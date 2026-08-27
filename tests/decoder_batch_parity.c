/*
 * decoder_batch_parity.c — does the CROSS-SLOT BATCHED streaming decoder produce the
 * same audio as the per-slot one?
 *
 * WHY THIS TEST EXISTS. `make test-serve-stream-batch` is an end-to-end gate: it
 * compares a streamed request against a single-stream reference and prints corr to
 * five decimals, so a small numeric drift would still read 1.00000. This one is the
 * microscope: same codes, same chunk schedule, same per-slot states, decoded BOTH
 * ways, compared sample by sample, and it prints the max absolute difference. The
 * expected answer is 0 (or, if it is not, a number small enough to be argued about
 * out loud — see the note on sgemm below).
 *
 * WHAT IT DELIBERATELY EXERCISES:
 *   - RAGGED batches. Every round gives each stream a DIFFERENT number of frames,
 *     which is the normal case under the ramped chunking, and the case a naive
 *     "batch only equal lengths" implementation would never reach.
 *   - streams that skip a round entirely (0 frames), i.e. a slot that has nothing
 *     pending while its neighbours decode.
 *   - the cold path (first chunk, all conv tails still zero) AND the warm path.
 *
 * WHAT WOULD FALSIFY THE BATCHING: any nonzero difference that grows with the number
 * of rounds. A per-item state that got shared would not show up as noise, it would
 * show up as one stream's audio leaking into another's — i.e. a LARGE difference,
 * localised in one stream. Read the per-stream lines, not only the max.
 *
 * NOTE ON EXACTNESS — measured on Apple M1 / Accelerate, 2026-08-18.
 * Most streams come out at exactly 0.000e+00. The residue that remains is NOT the
 * batching: it is the BLAS refusing to be shape-invariant, and it is reproducible with
 * a standalone sgemm probe that never touches this engine:
 *   - M == 1 goes through a GEMV kernel with a different K-reduction order than the
 *     M > 1 sgemm kernel: M=1 vs M=5, N=512, K=1024 -> 2.9e-05; M=2..16 vs 18 -> 0.
 *     This fires on the FIRST chunk of the ramp (one frame), in the pre-transformer.
 *   - N is not always neutral either: M=768, K=5376 gives exactly 0 for N=256, 262,
 *     274, 320 and 608 against N=608, but 2.4e-04 for N=310. 310 is exactly what the
 *     per-slot conv1d asks for (256 columns + a 54-column dilation-9 tail), which is
 *     why it shows up here and nowhere else.
 * Both are properties the PER-SLOT path already has: change a chunk size and its own
 * last bits move. Worst observed end-to-end here: 8.1e-06 with 1-frame chunks,
 * 3.1e-07 without — against a 16-bit LSB of 3.05e-05, i.e. below a quarter of one
 * output quantisation step.
 *
 * TWO REAL BUGS THIS TEST FOUND, both worth remembering because neither is arithmetic:
 * the depthwise conv and the final conv were first written as strided copies of the
 * per-slot loops. Identical source, different codegen — through an out-parameter the
 * compiler cannot prove non-aliasing and does not contract the k-loop into FMAs, while
 * in the per-slot version the output buffer is malloc'd locally and it does. One ULP,
 * amplified 480x by the upsampling stack into ~1e-05 on the audio. The fix was to make
 * both paths call literally the same function. **When two code paths must agree to the
 * last bit, share the function — do not copy the loop.**
 *
 * Build:  see the Makefile snippet in the task report, or by hand:
 *   gcc -Wall -O3 -march=native -ffast-math -Ivendor -DUSE_BLAS -DACCELERATE_NEW_LAPACK \
 *       -Ithird_party/ingot/include -o qwen_tts_batch_parity \
 *       tests/decoder_batch_parity.c $(ls *.o | grep -v main.o) vendor/lz4.o \
 *       third_party/ingot/libingot.a -lm -lpthread -framework Accelerate
 * Run:    ./qwen_tts_batch_parity [model_dir] [n_streams] [n_rounds] [codes.txt]
 */

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

/* Deterministic codes. Real dumped codes can be supplied instead (argv[4]); the
 * arithmetic under test does not care what the values are, only that BOTH paths see
 * exactly the same ones. */
static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

/* Ragged chunk schedule: stream s in round r gets a different frame count than its
 * neighbours, and sometimes zero. This is the shape the server actually produces.
 *
 * QWEN_PARITY_PAT=2 switches to a schedule with no 1-frame chunk. That is not
 * cosmetic, it ISOLATES the one thing that is genuinely not bit-exact: Accelerate
 * routes an sgemm with M==1 through a GEMV kernel whose K-reduction order differs
 * from the M>1 sgemm kernel. So a per-slot call on a ONE-frame chunk (the first chunk
 * of the ramp) and the same frame inside a multi-row batched call disagree in the last
 * bits — a property of the BLAS, not of this batching. Measured on this box with a
 * standalone sgemm probe: M=1 vs M=5, N=512, K=1024 -> 2.9e-05; M=3 vs M=5 -> exactly 0.
 * With PAT=2 the whole decoder is expected to come out bit-identical. */
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

    /* How many frames each stream will consume in total */
    int total_frames[MAX_STREAMS];
    int max_total = 0;
    for (int s = 0; s < n_streams; s++) {
        total_frames[s] = 0;
        for (int r = 0; r < n_rounds; r++) total_frames[s] += chunk_for(s, r);
        if (total_frames[s] > max_total) max_total = total_frames[s];
    }

    /* Load the model first: codebook_size bounds the codes we may generate. */
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
            /* Give each stream a different slice, wrapping if the dump is short. */
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

    /* ---- arm A: per-slot, one call per stream per round (the reference) ---- */
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

    /* ---- arm B: one batched call per round, ragged over the same streams ---- */
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

    /* ---- compare ---- */
    int fail = 0;
    double worst = 0.0;

    /* Split the verdict in two. The latent cache is the output of steps 1-5 (VQ,
     * pre-conv, pre-transformer, output proj); the audio is that plus the conv
     * decoder. Reporting only the audio would leave "which half drifted" a guess,
     * and the two halves have completely different suspects. */
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
