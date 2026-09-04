/* tests/roof_matvec_int8.c — primitive roof for the Talker decode B=1 int8 VNNI matvec.
 *
 * NOT a microbenchmark of a hot cache.  The question it answers is "how fast can this
 * CPU run THESE matvecs when the weights are cold", so the working set is built to be
 * the real one: 28 DISTINCT weight matrices per shape (one per Talker layer), cycled in
 * order exactly as one decode frame does.  That is 1.41 GB for the four shapes together,
 * ~22x the 32 MB L3 of one c8a CCX, so every weight byte comes from DRAM — the same
 * situation the serving engine is in, where the per-frame working set can never be
 * resident.
 *
 * `--matmat-bench` cannot answer this: its shapes are hard-coded to {3072x1024,
 * 1024x3072, 2048x1024} and it reuses ONE weight buffer, so at 25 MB the 12288x2048
 * case would sit in L3 and report a cache roof, not a DRAM roof.
 *
 * It also measures a READ-ONLY bandwidth roof, which tests/membw.c does not provide
 * (it has Copy and Triad only, both of which include write traffic).  A matvec is
 * read-dominated, so Triad is the wrong ceiling to compare against.
 *
 *   ./roof_matvec_int8 [--threads 8] [--layers 28] [--reps 5] [--json out.json]
 *
 * Run it under the same pinning as a prefork worker (taskset -c 0-7) so the topology
 * matches the serving worker being compared against.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "../qwen_tts_kernels.h"
#include "../qwen_tts_thread.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint64_t rng_s = 0x243F6A8885A308D3ull;
static float rf(void) {
    rng_s = rng_s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng_s >> 40) / (double)(1u << 24)) * 2.0f - 1.0f;
}

/* One projection of the Talker: `layers` distinct int8 weight matrices, so cycling
 * through them reproduces one frame's worth of cold weight streaming. */
typedef struct {
    const char *name;
    int rows, cols, layers;
    int8_t **W;
    float  **S;
    size_t  bytes;          /* weight bytes touched by one full cycle */
} proj_t;

static int build(proj_t *p, const char *name, int rows, int cols, int layers) {
    p->name = name; p->rows = rows; p->cols = cols; p->layers = layers;
    p->W = calloc(layers, sizeof *p->W);
    p->S = calloc(layers, sizeof *p->S);
    if (!p->W || !p->S) return -1;
    uint16_t *bf = malloc((size_t)rows * cols * sizeof(uint16_t));
    if (!bf) return -1;
    for (int l = 0; l < layers; l++) {
        p->W[l] = malloc((size_t)rows * cols);
        p->S[l] = malloc((size_t)rows * sizeof(float));
        if (!p->W[l] || !p->S[l]) { free(bf); return -1; }
        /* Different content per layer so nothing can be deduplicated by the allocator
         * or collapsed by the page cache. */
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            float v = rf(); uint32_t b; memcpy(&b, &v, 4);
            bf[i] = (uint16_t)((b + 0x8000u) >> 16);
        }
        qwen_quantize_bf16_to_int8(bf, rows, cols, p->W[l], p->S[l]);
    }
    free(bf);
    p->bytes = (size_t)rows * cols * layers;      /* 1 byte per int8 weight */
    return 0;
}

/* Read-only bandwidth roof: membw.c measures Copy and Triad, both of which move write
 * traffic too.  A B=1 matvec is a pure stream of reads, so this is the ceiling it
 * should actually be compared against. */
typedef struct { const uint64_t *buf; size_t n; uint64_t acc; } rd_job_t;
static rd_job_t g_rd[128];
static void read_task(size_t tid, size_t nt, void *ctx) {
    const rd_job_t *j = (const rd_job_t *)ctx;
    size_t per = j->n / nt, lo = tid * per, hi = (tid == nt - 1) ? j->n : lo + per;
    uint64_t a = 0;
    for (size_t i = lo; i < hi; i++) a += j->buf[i];
    g_rd[tid].acc = a;
}

static double read_roof_gbs(size_t mib, int nt, int reps) {
    size_t n = mib * 1024 * 1024 / sizeof(uint64_t);
    uint64_t *buf = malloc(n * sizeof(uint64_t));
    if (!buf) return -1.0;
    for (size_t i = 0; i < n; i++) buf[i] = i * 2654435761u;
    rd_job_t j = { buf, n, 0 };
    qwen_parallel((size_t)nt, read_task, &j);          /* warm */
    double best = 0.0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        qwen_parallel((size_t)nt, read_task, &j);
        double dt = now_ms() - t0;
        double gbs = (double)(n * sizeof(uint64_t)) / (dt * 1e6);
        if (gbs > best) best = gbs;
    }
    uint64_t sink = 0; for (int t = 0; t < nt; t++) sink += g_rd[t].acc;
    if (sink == 0x1234567) fprintf(stderr, "");   /* keep the reads */
    free(buf);
    return best;
}

int main(int argc, char **argv) {
    int nt = 8, layers = 28, reps = 5;
    const char *json = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--threads") && i + 1 < argc) nt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers")  && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")    && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")    && i + 1 < argc) json = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    qwen_set_threads(nt);
    qwen_threadpool_start(nt);

    /* The four shapes that the census says carry 100% of Talker decode, with the
     * serving call order inside a layer: qkv, o_proj, gate_up, down. */
    proj_t P[4];
    if (build(&P[0], "qkv_fused",  4096, 2048, layers) ||
        build(&P[1], "out_proj",   2048, 2048, layers) ||
        build(&P[2], "gate_up",   12288, 2048, layers) ||
        build(&P[3], "down",       2048, 6144, layers)) {
        fprintf(stderr, "allocation failed\n"); return 1;
    }
    size_t frame_bytes = 0;
    for (int i = 0; i < 4; i++) frame_bytes += P[i].bytes;

    int maxcols = 6144, maxrows = 12288;
    float *x = malloc((size_t)maxcols * sizeof(float));
    float *y = malloc((size_t)maxrows * sizeof(float));
    for (int i = 0; i < maxcols; i++) x[i] = rf();

    printf("primitive roof — Talker decode B=1 int8 VNNI, DRAM-resident\n");
    printf("  threads=%d  layers(distinct matrices)/shape=%d  reps=%d\n", nt, layers, reps);
    printf("  one full cycle over all four shapes = %.2f MB of weights"
           "  (one decode frame)\n\n", frame_bytes / 1048576.0);
    printf("  %-11s %6s x %-6s %9s %11s %11s\n",
           "shape", "rows", "cols", "MB/cycle", "ms/cycle", "GB/s");

    double gbs[4]; double ms[4];
    for (int i = 0; i < 4; i++) {
        proj_t *p = &P[i];
        for (int l = 0; l < p->layers; l++)                       /* warm */
            qwen_matvec_int8(y, p->W[l], p->S[l], x, p->rows, p->cols);
        double best = 1e18;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            for (int l = 0; l < p->layers; l++)
                qwen_matvec_int8(y, p->W[l], p->S[l], x, p->rows, p->cols);
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
        ms[i] = best;
        gbs[i] = (double)p->bytes / (best * 1e6);
        printf("  %-11s %6d x %-6d %9.1f %11.2f %11.1f\n",
               p->name, p->rows, p->cols, p->bytes / 1048576.0, best, gbs[i]);
    }

    /* The number that is directly comparable to the serving cost map: all four shapes
     * interleaved in the order one layer executes them, 28 layers = one frame. */
    for (int l = 0; l < layers; l++)
        for (int i = 0; i < 4; i++)
            qwen_matvec_int8(y, P[i].W[l], P[i].S[l], x, P[i].rows, P[i].cols);
    double fbest = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        for (int l = 0; l < layers; l++)
            for (int i = 0; i < 4; i++)
                qwen_matvec_int8(y, P[i].W[l], P[i].S[l], x, P[i].rows, P[i].cols);
        double dt = now_ms() - t0;
        if (dt < fbest) fbest = dt;
    }
    double fgbs = (double)frame_bytes / (fbest * 1e6);
    printf("  %-11s %6s   %-6s %9.1f %11.2f %11.1f   <- interleaved, real layer order\n",
           "FRAME(all4)", "-", "-", frame_bytes / 1048576.0, fbest, fgbs);

    double rd = read_roof_gbs(2048, nt, reps);
    printf("\n  read-only memory roof (%d threads, 2 GiB buffer): %.1f GB/s\n", nt, rd);
    printf("  (tests/membw.c reports Copy and Triad only; both include write traffic)\n");

    if (json) {
        FILE *f = fopen(json, "w");
        if (f) {
            fprintf(f, "{\n \"v\": 1,\n \"threads\": %d,\n \"layers_per_shape\": %d,\n"
                       " \"reps\": %d,\n \"frame_bytes\": %zu,\n"
                       " \"read_only_roof_gbs\": %.2f,\n \"shapes\": [\n", nt, layers, reps,
                    frame_bytes, rd);
            for (int i = 0; i < 4; i++)
                fprintf(f, "  { \"name\": \"%s\", \"rows\": %d, \"cols\": %d, "
                           "\"bytes_per_cycle\": %zu, \"ms_per_cycle\": %.4f, \"gbs\": %.2f }%s\n",
                        P[i].name, P[i].rows, P[i].cols, P[i].bytes, ms[i], gbs[i],
                        i == 3 ? "" : ",");
            fprintf(f, " ],\n \"frame\": { \"ms\": %.4f, \"gbs\": %.2f }\n}\n", fbest, fgbs);
            fclose(f);
            printf("  json: %s\n", json);
        }
    }
    return 0;
}
