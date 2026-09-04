/* membw.c — how much memory bandwidth this EXECUTION DOMAIN really has, and where it
 * saturates.
 *
 * "Execution domain", not "machine": on a multi-CCX x86 part the bandwidth a process can
 * reach depends on WHICH cpus it is allowed to run on, not on how many cores the host has.
 * Measured on an AWS c8a.4xlarge (2 CCX x 8 cores): the whole host triads at ~113 GB/s,
 * but a process confined to cpus 0-7 tops out at ~54 GB/s and one confined to 8-15 at
 * ~54 GB/s.  A worker roof can therefore NEVER be derived by dividing a host roof, on any
 * architecture: it has to be measured under the mask the work will actually run on.
 *
 * So this program reports the mask it ran under, taken from sched_getaffinity() — not the
 * count of online cpus, which is what it used to print and which silently stayed 16 under
 * `taskset -c 0-7`.
 *
 *   membw [--cpus 0-7] [--threads 1,2,4,8] [--reps 5] [--l3-mb 32] [--json] [--label L]
 *
 * Three kernels, because they do not measure the same ceiling:
 *   Copy   a[i]=b[i]              2 streams, read+write
 *   Triad  a[i]=b[i]+s*c[i]       3 streams, read+write
 *   Read   sum += b[i]            1 stream, READ ONLY  <- the ceiling a B=1 matvec sees
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <sched.h>
#endif

#define INNER 4
enum { K_COPY = 0, K_TRIAD = 1, K_READ = 2, K_N = 3 };
static const char *K_NAME[K_N] = { "copy", "triad", "read" };
/* streams touched per element, used to turn a duration into GB/s */
static const double K_STREAMS[K_N] = { 2.0, 3.0, 1.0 };

typedef struct {
    double *a, *b, *c;
    size_t  lo, hi;
    double  s;
    int     kernel;
    double  sink;
} job_t;

static void *worker(void *p) {
    job_t *j = (job_t *)p;
    for (int it = 0; it < INNER; it++) {
        if (j->kernel == K_COPY)
            for (size_t i = j->lo; i < j->hi; i++) j->a[i] = j->b[i];
        else if (j->kernel == K_TRIAD)
            for (size_t i = j->lo; i < j->hi; i++) j->a[i] = j->b[i] + j->s * j->c[i];
        else {
            double acc = 0.0;
            for (size_t i = j->lo; i < j->hi; i++) acc += j->b[i];
            j->sink += acc;          /* kept and printed on demand, so it cannot be elided */
        }
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

/* ---- affinity: the mask is the identity of the measurement --------------------- */

#if defined(__APPLE__)
/* macOS has no sched_setaffinity and no stable way to confine a process to a core set;
 * the mask is reported as "all" and --cpus is refused rather than silently ignored. */
static int mask_apply(const char *spec) { (void)spec; return -1; }
static int mask_count(void) {
    int n = 1; size_t len = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &len, NULL, 0) != 0)
        n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
}
static void mask_string(char *out, size_t cap) { snprintf(out, cap, "all"); }
#else
static int mask_apply(const char *spec) {
    cpu_set_t set; CPU_ZERO(&set);
    char buf[512]; snprintf(buf, sizeof buf, "%s", spec);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int a, b;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) { for (int c = a; c <= b; c++) CPU_SET(c, &set); }
        else if (sscanf(tok, "%d", &a) == 1)   { CPU_SET(a, &set); }
        else return -1;
    }
    if (CPU_COUNT(&set) == 0) return -1;
    return sched_setaffinity(0, sizeof set, &set);
}
static int mask_count(void) {
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        return n > 0 ? (int)n : 1;
    }
    int n = CPU_COUNT(&set);
    return n > 0 ? n : 1;
}
/* "0-7", "0-3,8-11", "2" — the same compact form /proc/<pid>/status Cpus_allowed_list uses,
 * so a mask measured here compares literally with a mask observed on a running worker. */
static void mask_string(char *out, size_t cap) {
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) { snprintf(out, cap, "?"); return; }
    size_t o = 0; out[0] = '\0';
    int i = 0, ncpu = (int)sysconf(_SC_NPROCESSORS_CONF);
    if (ncpu <= 0 || ncpu > CPU_SETSIZE) ncpu = CPU_SETSIZE;
    while (i < ncpu) {
        if (!CPU_ISSET(i, &set)) { i++; continue; }
        int j = i;
        while (j + 1 < ncpu && CPU_ISSET(j + 1, &set)) j++;
        int k = (i == j) ? snprintf(out + o, cap - o, "%s%d", o ? "," : "", i)
                         : snprintf(out + o, cap - o, "%s%d-%d", o ? "," : "", i, j);
        if (k < 0 || (size_t)k >= cap - o) break;
        o += (size_t)k;
        i = j + 1;
    }
    if (!o) snprintf(out, cap, "?");
}
#endif

static double run_once(double *a, double *b, double *c, size_t n, int nt, int kernel,
                       double *sink) {
    pthread_t th[256];
    job_t     jb[256];
    if (nt > 256) nt = 256;
    size_t chunk = (n + (size_t)nt - 1) / (size_t)nt;
    for (int t = 0; t < nt; t++) {
        jb[t].a = a; jb[t].b = b; jb[t].c = c; jb[t].s = 3.0; jb[t].kernel = kernel;
        jb[t].sink = 0.0;
        jb[t].lo = (size_t)t * chunk;
        jb[t].hi = jb[t].lo + chunk; if (jb[t].hi > n) jb[t].hi = n;
        if (jb[t].lo > n) jb[t].lo = n;
    }
    double t0 = now_s();
    for (int t = 0; t < nt; t++) pthread_create(&th[t], NULL, worker, &jb[t]);
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    double dt = now_s() - t0;
    for (int t = 0; t < nt; t++) *sink += jb[t].sink;
    return dt;
}

/* Smallest thread count in the sweep that reaches `frac` of this kernel's own peak.
 * Kept as a curve and not collapsed into one number: on the c8a's cpus 0-7 two threads
 * already deliver ~97% of the local roof, which is a topology fact, not a footnote. */
static int t_at(const double *g, const int *ts, int nts, double peak, double frac) {
    for (int i = 0; i < nts; i++) if (g[i] >= frac * peak) return ts[i];
    return ts[nts - 1];
}

int main(int argc, char **argv) {
    int    l3_mb = 32, reps = 5, json = 0;
    const char *tlist = NULL, *label = "", *cpus = NULL;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--l3-mb")   && i + 1 < argc) l3_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps")    && i + 1 < argc) reps  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) tlist = argv[++i];
        else if (!strcmp(argv[i], "--label")   && i + 1 < argc) label = argv[++i];
        else if (!strcmp(argv[i], "--cpus")    && i + 1 < argc) cpus  = argv[++i];
        else if (!strcmp(argv[i], "--json")) json = 1;
        else { fprintf(stderr, "membw: opzione sconosciuta '%s'\n", argv[i]); return 2; }
    }
    if (l3_mb < 1) l3_mb = 32;
    if (reps  < 1) reps  = 1;

    if (cpus && mask_apply(cpus) != 0) {
        fprintf(stderr, "membw: cannot apply --cpus %s on this platform/mask\n", cpus);
        return 2;
    }
    int ncpu = mask_count();               /* CPUs THIS PROCESS MAY USE, not online cpus */
    char maskstr[512]; mask_string(maskstr, sizeof maskstr);

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

    size_t per_mb = (size_t)l3_mb * 4;
    if (per_mb < 64)  per_mb = 64;
    if (per_mb > 512) per_mb = 512;
    size_t n = per_mb * 1024u * 1024u / sizeof(double);

    double *a = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "membw: malloc fallita (%zu MiB x3)\n", per_mb); return 1; }

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

    double g[K_N][16];
    double sink = 0.0;
    for (int i = 0; i < nts; i++)
        for (int k = 0; k < K_N; k++) {
            double best = 0.0;
            for (int r = 0; r < reps; r++) {
                double dt = run_once(a, b, c, n, ts[i], k, &sink);
                double gbs = (double)INNER * K_STREAMS[k] * (double)n * sizeof(double) / dt / 1e9;
                if (gbs > best) best = gbs;
            }
            g[k][i] = best;
        }

    double peak[K_N]; int peak_t[K_N], t90[K_N], t95[K_N], t99[K_N];
    for (int k = 0; k < K_N; k++) {
        peak[k] = 0.0; peak_t[k] = ts[0];
        for (int i = 0; i < nts; i++) if (g[k][i] > peak[k]) { peak[k] = g[k][i]; peak_t[k] = ts[i]; }
        t90[k] = t_at(g[k], ts, nts, peak[k], 0.90);
        t95[k] = t_at(g[k], ts, nts, peak[k], 0.95);
        t99[k] = t_at(g[k], ts, nts, peak[k], 0.99);
    }
    /* The working set is sized to >= 4x the LLC it was told about, so every kernel here is
     * DRAM-resident by construction; a cache-resident number would be a different roof. */
    const char *residency = ((double)per_mb >= 4.0 * l3_mb) ? "dram" : "cache";

    if (json) {
        printf("{\"kind\":\"membw\",\"v\":2,\"label\":\"%s\",\"array_mib_per_buffer\":%zu,"
               "\"total_mib\":%zu,\"working_set_mib\":%zu,\"residency\":\"%s\","
               "\"dtype\":\"double\",\"reps\":%d,\"cpus_seen\":%d,"
               "\"cpu_mask\":\"%s\",\"cpu_count\":%d,\"l3_mb_assumed\":%d,\"sweep\":[",
               label, per_mb, per_mb * 3, per_mb * 3, residency, reps, ncpu,
               maskstr, ncpu, l3_mb);
        for (int i = 0; i < nts; i++)
            printf("%s{\"threads\":%d,\"copy_gbs\":%.2f,\"triad_gbs\":%.2f,\"read_gbs\":%.2f}",
                   i ? "," : "", ts[i], g[K_COPY][i], g[K_TRIAD][i], g[K_READ][i]);
        printf("],");
        for (int k = 0; k < K_N; k++)
            printf("\"%s\":{\"peak_gbs\":%.2f,\"peak_threads\":%d,\"t90\":%d,\"t95\":%d,"
                   "\"t99\":%d,\"streams\":%.0f,\"access\":\"%s\"},",
                   K_NAME[k], peak[k], peak_t[k], t90[k], t95[k], t99[k], K_STREAMS[k],
                   k == K_READ ? "read-only" : "read-write");
        /* v1 keys kept so tools/box_info.sh and every archived parser keep working */
        printf("\"peak_triad_gbs\":%.2f,\"peak_triad_threads\":%d,\"knee_threads\":%d}\n",
               peak[K_TRIAD], peak_t[K_TRIAD], t95[K_TRIAD]);
    } else {
        printf("membw%s%s%s — mask %s (%d cpu allowed), array %zu MiB x3 (>= 4x L3 di %d MiB),"
               " %s-resident, best of %d\n",
               label[0] ? " [" : "", label, label[0] ? "]" : "", maskstr, ncpu, per_mb,
               l3_mb, residency, reps);
        printf("  %-8s %12s %12s %12s\n", "thread", "Copy GB/s", "Triad GB/s", "Read GB/s");
        for (int i = 0; i < nts; i++)
            printf("  %-8d %12.1f %12.1f %12.1f\n", ts[i], g[K_COPY][i], g[K_TRIAD][i], g[K_READ][i]);
        for (int k = 0; k < K_N; k++)
            printf("  %-5s peak %6.1f GB/s @%2dT   90%% @%2dT   95%% @%2dT   99%% @%2dT\n",
                   K_NAME[k], peak[k], peak_t[k], t90[k], t95[k], t99[k]);
        printf("  Read is the ceiling a B=1 matvec sees; Copy/Triad include write traffic.\n");
        printf("  This is the roof of MASK %s only. Never divide a host roof to get a worker roof.\n",
               maskstr);
    }
    if (sink == 12345.6789) fprintf(stderr, " ");   /* keep the read kernel */
    free(a); free(b); free(c);
    return 0;
}
