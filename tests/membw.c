/* membw.c — how much memory bandwidth this machine REALLY has, and where it saturates.
 *
 * WHY IT EXISTS, and why the datasheet figure is not enough. This engine's bottleneck is
 * not the ALU: the Code Predictor re-reads its weights 16 TIMES PER FRAME. So the question
 * that decides batching is not "how many GFLOP/s" but:
 *
 *     at how many threads does bandwidth stop rising?
 *
 * Because that is the fork: if bandwidth saturates at 2 threads, giving 8 threads to ONE
 * request is wasted and 2 threads to FOUR requests is better; if it scales to every core,
 * the opposite holds. Without this number the thread x batch matrix is explored by
 * guesswork, and on a machine rented by the hour guesswork is expensive.
 *
 * WHAT IT IS. STREAM cut to the bone: Copy (a=b) and Triad (a=b+s*c) over three double
 * arrays, with a thread sweep and best-of-N. A few seconds, not a paper benchmark.
 *
 * THE TWO THINGS THAT WOULD MAKE IT A LIE, both handled here:
 *   1. ARRAYS TOO SMALL -> it measures cache rather than DRAM and the number comes out
 *      5-10x too high. The arrays are sized to at least 4x the L3 (--l3-mb, passed by the
 *      box-info script, which actually reads the L3).
 *   2. SEQUENTIAL FIRST TOUCH -> on a NUMA machine every page lands on the initialising
 *      thread's node, and the test then measures only that node forever. Here the
 *      initialisation is PARALLEL and uses the same partition as the measurement, so pages
 *      land where the thread that reads them is running.
 *
 * Use:
 *   membw [--l3-mb N] [--threads LIST] [--reps N] [--json] [--label TEXT]
 *     --l3-mb N     size the arrays to max(4*N, 64) MiB each (default 32)
 *     --threads     "1,2,4,8" (default: 1, 2, 4, half the cores, all the cores)
 *     --reps N      runs per cell, the BEST is kept (default 5)
 *     --json        one JSON line, to be embedded in the machine's report
 *     --label TEXT  free-form label (used for numactl cells: "numa-local"/"numa-cross")
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

/* no pthread_barrier on macOS: threads are created per timed run, with enough inner
 * iterations that the create/join cost is noise (<1 %). */
#define INNER 4

typedef struct {
    double *a, *b, *c;
    size_t  lo, hi;
    double  s;
    int     kernel;          /* 0 = copy, 1 = triad */
} job_t;

static void *worker(void *p) {
    job_t *j = (job_t *)p;
    for (int it = 0; it < INNER; it++) {
        if (j->kernel == 0) for (size_t i = j->lo; i < j->hi; i++) j->a[i] = j->b[i];
        else                for (size_t i = j->lo; i < j->hi; i++) j->a[i] = j->b[i] + j->s * j->c[i];
    }
    return NULL;
}

static void *initer(void *p) {
    job_t *j = (job_t *)p;
    for (size_t i = j->lo; i < j->hi; i++) { j->a[i] = 1.0; j->b[i] = 2.0; j->c[i] = 0.5; }
    return NULL;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int n_cpus(void) {
#if defined(__APPLE__)
    int n = 1; size_t len = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &len, NULL, 0) != 0) n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/* un giro cronometrato: nt thread, INNER passate, ritorna i secondi */
static double run_once(double *a, double *b, double *c, size_t n, int nt, int kernel) {
    pthread_t th[256];
    job_t     jb[256];
    if (nt > 256) nt = 256;
    size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;
    for (int t = 0; t < nt; t++) {
        jb[t].a = a; jb[t].b = b; jb[t].c = c; jb[t].s = 3.0; jb[t].kernel = kernel;
        jb[t].lo = (size_t)t * chunk;
        jb[t].hi = jb[t].lo + chunk; if (jb[t].hi > n) jb[t].hi = n;
        if (jb[t].lo > n) jb[t].lo = n;
    }
    double t0 = now_s();
    for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, worker, &jb[t]);
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    return now_s() - t0;
}

int main(int argc, char **argv) {
    int    l3_mb = 32, reps = 5, json = 0;
    const char *tlist = NULL, *label = "";
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--l3-mb")   && i + 1 < argc) l3_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")    && i + 1 < argc) reps  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) tlist = argv[++i];
        else if (!strcmp(argv[i], "--label")   && i + 1 < argc) label = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else { fprintf(stderr, "membw: opzione sconosciuta '%s'\n", argv[i]); return 2; }
    }
    if (l3_mb < 1) l3_mb = 32;
    if (reps  < 1) reps  = 1;

    int ncpu = n_cpus();

    /* default sweep: 1, 2, 4, half the cores, all of them. The knee is almost always
     * between 2 and "half", which is exactly the point that decides threads per request. */
    int  ts[16], nts = 0;
    if (tlist) {
        char buf[256]; snprintf(buf, sizeof buf, "%s", tlist);
        for (char *tok = strtok(buf, ","); tok && nts < 16; tok = strtok(NULL, ",")) {
            int v = atoi(tok); if (v > 0) ts[nts++] = v;
        }
    } else {
        int cand[5] = { 1, 2, 4, ncpu / 2, ncpu };
        for (int i = 0; i < 5; i++) {
            if (cand[i] < 1 || cand[i] > ncpu) continue;
            int dup = 0; for (int k = 0; k < nts; k++) if (ts[k] == cand[i]) dup = 1;
            if (!dup) ts[nts++] = cand[i];
        }
    }
    if (nts == 0) ts[nts++] = 1;

    /* >= 4x la L3 per array, cosi' nessun riuso di cache puo' gonfiare il numero.
     * Tetto a 512 MiB per array: oltre non si misura di piu', si aspetta e basta. */
    size_t per_mb = (size_t)l3_mb * 4;
    if (per_mb < 64)  per_mb = 64;
    if (per_mb > 512) per_mb = 512;
    size_t n = per_mb * 1024u * 1024u / sizeof(double);

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "membw: malloc fallita (%zu MiB x3)\n", per_mb); return 1; }

    /* first-touch PARALLELO, con la stessa partizione della misura (vedi testata) */
    {
        pthread_t th[256]; job_t jb[256];
        int nt = ts[nts - 1] > 256 ? 256 : ts[nts - 1];
        size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;
        for (int t = 0; t < nt; t++) {
            jb[t].a = a; jb[t].b = b; jb[t].c = c;
            jb[t].lo = (size_t)t * chunk;
            jb[t].hi = jb[t].lo + chunk; if (jb[t].hi > n) jb[t].hi = n;
            if (jb[t].lo > n) jb[t].lo = n;
            pthread_create(&th[t], NULL, initer, &jb[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    }

    double copy_gbs[16], triad_gbs[16];
    for (int i = 0; i < nts; i++) {
        double best_c = 0.0, best_t = 0.0;
        for (int r = 0; r < reps; r++) {
            double dt = run_once(a, b, c, n, ts[i], 0);
            /* Copy tocca 2 array (1 letto + 1 scritto) per iterazione interna */
            double gbs = (double)INNER * 2.0 * (double)n * sizeof(double) / dt / 1e9;
            if (gbs > best_c) best_c = gbs;
        }
        for (int r = 0; r < reps; r++) {
            double dt = run_once(a, b, c, n, ts[i], 1);
            /* Triad ne tocca 3 (2 letti + 1 scritto) */
            double gbs = (double)INNER * 3.0 * (double)n * sizeof(double) / dt / 1e9;
            if (gbs > best_t) best_t = gbs;
        }
        copy_gbs[i] = best_c; triad_gbs[i] = best_t;
    }

    /* IL numero che serve: il picco, e il PRIMO conteggio di thread che arriva al 95%
     * del picco. Quello e' il ginocchio — oltre, i thread in piu' non comprano banda. */
    double peak = 0.0; int peak_t = ts[0];
    for (int i = 0; i < nts; i++) if (triad_gbs[i] > peak) { peak = triad_gbs[i]; peak_t = ts[i]; }
    int knee = peak_t;
    for (int i = 0; i < nts; i++) if (triad_gbs[i] >= 0.95 * peak) { knee = ts[i]; break; }

    if (json) {
        printf("{\"kind\":\"membw\",\"label\":\"%s\",\"array_mib_per_buffer\":%zu,"
               "\"total_mib\":%zu,\"dtype\":\"double\",\"reps\":%d,\"cpus_seen\":%d,\"sweep\":[",
               label, per_mb, per_mb * 3, reps, ncpu);
        for (int i = 0; i < nts; i++)
            printf("%s{\"threads\":%d,\"copy_gbs\":%.2f,\"triad_gbs\":%.2f}",
                   i ? "," : "", ts[i], copy_gbs[i], triad_gbs[i]);
        printf("],\"peak_triad_gbs\":%.2f,\"peak_triad_threads\":%d,\"knee_threads\":%d}\n",
               peak, peak_t, knee);
    } else {
        printf("membw%s%s%s — array %zu MiB x3 (>= 4x L3 di %d MiB), best of %d, %d cpu viste\n",
               label[0] ? " [" : "", label, label[0] ? "]" : "", per_mb, l3_mb, reps, ncpu);
        printf("  %-8s %12s %12s\n", "thread", "Copy GB/s", "Triad GB/s");
        for (int i = 0; i < nts; i++)
            printf("  %-8d %12.1f %12.1f\n", ts[i], copy_gbs[i], triad_gbs[i]);
        printf("  picco Triad %.1f GB/s a %d thread; GINOCCHIO a %d thread (95%% del picco)\n",
               peak, peak_t, knee);
        printf("  -> oltre %d thread la banda non sale: conviene dare %d thread a PIU' richieste\n"
               "     invece che tutti i core a una sola (il CP rilegge i pesi 16x per frame).\n",
               knee, knee);
    }
    free(a); free(b); free(c);
    return 0;
}
