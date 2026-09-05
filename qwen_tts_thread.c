/* qwen_tts_thread.c - Cross-OS parallel-for */
#include "qwen_tts_thread.h"
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <time.h>

double qwen_parallel_now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* Sense-reversing spin barrier for phases inside one region: every participant is a
 * distinct thread already running (the region was dispatched with nt == team), and the
 * phases are microseconds long, so spinning is the right wait. */
void qwen_barrier_init(qwen_barrier_t *b, int nt) { b->arrived = 0; b->phase = 0; b->nt = nt; }
void qwen_barrier_wait(qwen_barrier_t *b) {
    int ph = __atomic_load_n(&b->phase, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&b->arrived, 1, __ATOMIC_ACQ_REL) == b->nt) {
        __atomic_store_n(&b->arrived, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&b->phase, ph + 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&b->phase, __ATOMIC_ACQUIRE) == ph) {
#if defined(__x86_64__) || defined(__i386__)
            __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
#endif
        }
    }
}
#include "qwen_tts_costmap.h"

static __thread int g_qwen_tls_tag = 0;
int  qwen_tls_tag_get(void) { return g_qwen_tls_tag; }
void qwen_tls_tag_set(int tag) { g_qwen_tls_tag = tag; }
#include "qwen_tts_kernels.h"

#include <stdatomic.h>
#include <time.h>

static _Atomic long long g_qp_busy_us, g_qp_chunks, g_qp_dispatches;
static int g_qp_meter;

void qwen_parallel_meter(int on) {
    g_qp_meter = on;
    if (on) { atomic_store(&g_qp_busy_us, 0); atomic_store(&g_qp_chunks, 0);
              atomic_store(&g_qp_dispatches, 0); }
}
void qwen_parallel_meter_read(double *busy_ms, long long *chunks, long long *dispatches) {
    if (busy_ms)   *busy_ms   = (double)atomic_load(&g_qp_busy_us) / 1e3;
    if (chunks)    *chunks    = atomic_load(&g_qp_chunks);
    if (dispatches)*dispatches= atomic_load(&g_qp_dispatches);
}
__attribute__((unused))
static inline long long qp_now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000LL + t.tv_nsec / 1000;
}

#if defined(__APPLE__) && defined(__BLOCKS__) && !defined(QWEN_FORCE_PTHREAD)

#include <dispatch/dispatch.h>

static __thread int g_qp_depth;

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx) {
    if (nt == 0) return;
    if (nt == 1) { g_qp_depth++; fn(0, 1, ctx); g_qp_depth--; return; }
    const int tag = g_qwen_tls_tag;
    /* GCD's dispatch_apply both dispatches and joins, so on this backend the wait is
     * inside the call and NOT separable: pool_wait_completion stays unrecorded here
     * rather than being filled with a guess. */
    qwen_region_begin(QWEN_RGN_RT_POOL_DISPATCH);
    dispatch_apply(nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t tid) {
        qwen_ftz_on();
        qwen_tls_tag_set(tag);
        g_qp_depth++;
        fn(tid, nt, ctx);
        g_qp_depth--;
    });
    qwen_region_end(QWEN_RGN_RT_POOL_DISPATCH);
}

void qwen_threadpool_start(int n_threads) { (void)n_threads; }
void qwen_threadpool_stop(void) {}
void qwen_threadpool_after_fork(void) {}
/* dispatch_apply re-enters and accepts concurrent submitters; it owns no job slot.
 * It exposes no submit priority, so set_low_until below is a no-op and says so. */
int qwen_pool_nested_dispatch_ok(void) { return 1; }
int qwen_pool_concurrent_submit_ok(void) { return 1; }
int qwen_pool_priority_ok(void) { return 0; }
/* Truthful now: a task running under dispatch_apply IS inside the pool, so a nested
 * qwen_parallel caller runs its body inline instead of dispatching a second apply on top
 * of the first.  Reporting 0 here (as this did) was safe but oversubscribed. */
int qwen_parallel_active(void) { return g_qp_depth > 0; }
void qwen_parallel_set_low_until(double until_ms) { (void)until_ms; }
/* 0 = this backend cannot hold a fixed team of workers inside a spin barrier, so the
 * persistent regions stay off here.  It is a capability answer, not a thread count. */
int qwen_parallel_team(void) { return 0; }

#elif defined(_WIN32) && !defined(QWEN_USE_PTHREADS)

#include <windows.h>

typedef struct {
    qwen_task_fn fn;
    void *ctx;
    size_t nt;
    volatile LONG64 next;
    int tag;
} qwen_job_t;

static struct {
    HANDLE *threads;
    int nworkers;
    CRITICAL_SECTION mtx;
    CONDITION_VARIABLE wake;
    CONDITION_VARIABLE complete;
    qwen_job_t *job;
    unsigned long generation;
    int completed;
    int stop;
} P;
static int g_inited = 0;

static __thread int g_qp_depth;

static void run_chunks(qwen_job_t *job) {
    LONG64 v;
    while ((v = InterlockedIncrement64(&job->next)) <= (LONG64)job->nt) {
        qwen_tls_tag_set(job->tag);
        g_qp_depth++;
        job->fn((size_t)(v - 1), job->nt, job->ctx);
        g_qp_depth--;
    }
}

static DWORD WINAPI worker_main(LPVOID arg) {
    qwen_ftz_on();
    EnterCriticalSection(&P.mtx);
    unsigned long seen = (unsigned long)(ULONG_PTR)arg;
    for (;;) {
        while (!P.stop && P.generation == seen)
            SleepConditionVariableCS(&P.wake, &P.mtx, INFINITE);
        if (P.stop) break;
        seen = P.generation;
        qwen_job_t *job = P.job;
        LeaveCriticalSection(&P.mtx);
        if (job) run_chunks(job);
        EnterCriticalSection(&P.mtx);
        if (++P.completed == P.nworkers)
            WakeConditionVariable(&P.complete);
    }
    LeaveCriticalSection(&P.mtx);
    return 0;
}

void qwen_threadpool_stop(void) {
    if (!g_inited || !P.threads) return;
    EnterCriticalSection(&P.mtx);
    P.stop = 1;
    WakeAllConditionVariable(&P.wake);
    LeaveCriticalSection(&P.mtx);
    for (int i = 0; i < P.nworkers; i++) {
        WaitForSingleObject(P.threads[i], INFINITE);
        CloseHandle(P.threads[i]);
    }
    free(P.threads);
    P.threads = NULL;
    P.nworkers = 0;
    P.stop = 0;
}

void qwen_threadpool_after_fork(void) {}

void qwen_threadpool_start(int n_threads) {
    int want = n_threads > 1 ? n_threads - 1 : 0;
    if (!g_inited) {
        InitializeCriticalSection(&P.mtx);
        InitializeConditionVariable(&P.wake);
        InitializeConditionVariable(&P.complete);
        P.generation = 0;
        g_inited = 1;
    }
    if (want == P.nworkers) return;
    qwen_threadpool_stop();
    if (want == 0) return;
    P.threads = (HANDLE *)malloc(sizeof(HANDLE) * (size_t)want);
    if (!P.threads) { P.nworkers = 0; return; }
    int created = 0;
    unsigned long gen0 = P.generation;
    for (int i = 0; i < want; i++) {
        P.threads[i] = CreateThread(NULL, 0, worker_main,
                                    (LPVOID)(ULONG_PTR)gen0, 0, NULL);
        if (!P.threads[i]) break;
        created++;
    }
    P.nworkers = created;
    if (created == 0) { free(P.threads); P.threads = NULL; }
}

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx) {
    if (nt == 0) return;
    if (!g_inited || P.nworkers == 0 || nt == 1) {
        for (size_t i = 0; i < nt; i++) fn(i, nt, ctx);
        return;
    }
    qwen_job_t job;
    job.fn = fn; job.ctx = ctx; job.nt = nt; job.next = 0; job.tag = g_qwen_tls_tag;
    EnterCriticalSection(&P.mtx);
    P.job = &job;
    P.completed = 0;
    P.generation++;
    WakeAllConditionVariable(&P.wake);
    LeaveCriticalSection(&P.mtx);

    run_chunks(&job);

    EnterCriticalSection(&P.mtx);
    while (P.completed != P.nworkers)
        SleepConditionVariableCS(&P.complete, &P.mtx, INFINITE);
    P.job = NULL;
    LeaveCriticalSection(&P.mtx);
}

/* One job slot and no submit lock held across the dispatch: a nested call deadlocks and
 * two submitters would overwrite each other's job.  No submit priority either. */
int qwen_pool_nested_dispatch_ok(void) { return 0; }
int qwen_pool_concurrent_submit_ok(void) { return 0; }
int qwen_pool_priority_ok(void) { return 0; }
/* The single global job slot means a nested qwen_parallel would deadlock; reporting the
 * depth truthfully makes every in-pool caller run inline instead. */
int qwen_parallel_active(void) { return g_qp_depth > 0; }
void qwen_parallel_set_low_until(double until_ms) { (void)until_ms; }
int qwen_parallel_team(void) { return 0; }

#else

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    qwen_task_fn fn;
    void *ctx;
    size_t nt;
    atomic_size_t next;
    int tag;
} qwen_job_t;

typedef struct { unsigned long seen0; int idx; } qwen_worker_arg_t;

#define QWEN_GW_NEED_BITS 16
#define QWEN_GW_NEED_MASK ((1UL << QWEN_GW_NEED_BITS) - 1UL)
#define QWEN_GW_NEED(w)   ((int)((unsigned long)(w) & QWEN_GW_NEED_MASK))
#define QWEN_GW_GEN(w)    ((unsigned long)(w) >> QWEN_GW_NEED_BITS)
#define QWEN_GW_MAKE(g,n) (((unsigned long)(g) << QWEN_GW_NEED_BITS) | \
                           ((unsigned long)(n) & QWEN_GW_NEED_MASK))

static struct {
    pthread_t *threads;
    qwen_worker_arg_t *wargs;
    int nworkers;
    pthread_mutex_t submit_mtx;
    pthread_mutex_t mtx;
    pthread_cond_t wake;
    pthread_cond_t complete;
    qwen_job_t *job;
    _Atomic unsigned long generation;
    _Atomic int completed;
    int sleeping;
    unsigned long long sleep_mask;
    int sleeping_hi;
    int main_sleeping;
    _Atomic int stop;
} P;
static int g_inited = 0;

#if defined(__linux__) && defined(__aarch64__)
#define QWEN_POOL_SPIN_DEFAULT 65536
#else
#define QWEN_POOL_SPIN_DEFAULT 4096
#endif
static int qwen_pool_spin(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_POOL_SPIN");
        v = e ? atoi(e) : QWEN_POOL_SPIN_DEFAULT;
        if (v < 0) v = 0;
    }
    return v;
}
static int qwen_pool_narrow(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_POOL_NARROW");
        v = (e && e[0] == '1');
    }
    return v;
}
int qwen_pool_spin_value(void)   { return qwen_pool_spin(); }
int qwen_pool_narrow_value(void) { return qwen_pool_narrow(); }

static inline void qwen_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

#ifdef QWEN_POOL_STATS
#include <stdio.h>
static _Atomic unsigned long ps_dispatch, ps_chunks, ps_worker_park, ps_main_park, ps_serial;
#define PS_INC(c) atomic_fetch_add_explicit(&(c), 1, memory_order_relaxed)
static void ps_report_body(void);
static void ps_report(void);
__attribute__((constructor)) static void ps_register(void) { atexit(ps_report); }
__attribute__((destructor)) static void ps_report_dtor(void) { ps_report(); }
static void ps_report(void) {
    static _Atomic int done = 0;
    if (atomic_exchange_explicit(&done, 1, memory_order_relaxed)) return;
    ps_report_body();
}
void qwen_pool_stats_report(void) { ps_report(); }
static void ps_report_body(void) {
    fprintf(stderr, "POOLSTATS dispatch=%lu chunks=%lu worker_park=%lu main_park=%lu serial=%lu\n",
            ps_dispatch, ps_chunks, ps_worker_park, ps_main_park, ps_serial);
}
#else
void qwen_pool_stats_report(void) { }
#define PS_INC(c) ((void)0)
#endif

static __thread int g_qp_depth = 0;
int qwen_parallel_active(void) { return g_qp_depth > 0; }

static __thread double g_qp_low_until = 0.0;
static _Atomic double g_qp_last_hi_ms = 0.0;   /* when a HIGH submitter last dispatched */
void qwen_parallel_set_low_until(double until_ms) { g_qp_low_until = until_ms; }
static double qwen_pool_hi_window_ms(void) {
    static double v = -1.0;
    if (v < 0) { const char *e = getenv("QWEN_POOL_HI_WINDOW_US"); v = (e ? atof(e) : 200.0) / 1000.0; }
    return v;
}

static void run_chunks(qwen_job_t *job) {
    size_t i;
    qwen_tls_tag_set(job->tag);
    g_qp_depth++;
    while ((i = atomic_fetch_add(&job->next, 1)) < job->nt)
    {   PS_INC(ps_chunks);
        if (g_qp_meter) {
            long long t0 = qp_now_us();
            job->fn(i, job->nt, job->ctx);
            atomic_fetch_add(&g_qp_busy_us, qp_now_us() - t0);
            atomic_fetch_add(&g_qp_chunks, 1);
        } else job->fn(i, job->nt, job->ctx); }
    g_qp_depth--;
}

static void *worker_main(void *arg) {
    qwen_ftz_on();
    const qwen_worker_arg_t *wa = (const qwen_worker_arg_t *)arg;
    { char name[16]; snprintf(name, sizeof name, "qwen-pool-%d", wa->idx);
#if defined(__linux__)
      prctl(PR_SET_NAME, name, 0, 0, 0);
#endif
    }
    const int my_idx = wa->idx;
    const unsigned long long my_bit = my_idx < 64 ? (1ULL << my_idx) : 0ULL;
    unsigned long seen = wa->seen0;
    for (;;) {
        unsigned long gw = atomic_load_explicit(&P.generation, memory_order_acquire);
        int budget = qwen_pool_spin();
        while (budget-- > 0 && !P.stop &&
               !(gw != seen && QWEN_GW_NEED(gw) > my_idx)) {
            qwen_cpu_relax();
            gw = atomic_load_explicit(&P.generation, memory_order_acquire);
        }

        if (!P.stop && !(gw != seen && QWEN_GW_NEED(gw) > my_idx)) {
            pthread_mutex_lock(&P.mtx);
            P.sleeping++;
            if (my_bit) P.sleep_mask |= my_bit; else P.sleeping_hi++;
            PS_INC(ps_worker_park);
            for (;;) {
                gw = atomic_load_explicit(&P.generation, memory_order_relaxed);
                if (P.stop || (gw != seen && QWEN_GW_NEED(gw) > my_idx)) break;
                pthread_cond_wait(&P.wake, &P.mtx);
            }
            if (my_bit) P.sleep_mask &= ~my_bit; else P.sleeping_hi--;
            P.sleeping--;
            pthread_mutex_unlock(&P.mtx);
        }
        if (P.stop) break;

        seen = atomic_load_explicit(&P.generation, memory_order_acquire);
        const int need = QWEN_GW_NEED(seen);
        qwen_job_t *job = P.job;
        if (job && my_idx < need) {
            run_chunks(job);
            if (atomic_fetch_add_explicit(&P.completed, 1, memory_order_acq_rel) + 1
                    == need) {
                pthread_mutex_lock(&P.mtx);
                if (P.main_sleeping) pthread_cond_signal(&P.complete);
                pthread_mutex_unlock(&P.mtx);
            }
        }
    }
    return NULL;
}

void qwen_threadpool_stop(void) {
    if (!g_inited || !P.threads) return;
    pthread_mutex_lock(&P.mtx);
    P.stop = 1;
    /* Bump generation too: a spinning worker must observe stop, not spin forever. */
    {
        unsigned long gw = atomic_load_explicit(&P.generation, memory_order_relaxed);
        atomic_store_explicit(&P.generation,
                              QWEN_GW_MAKE(QWEN_GW_GEN(gw) + 1, QWEN_GW_NEED_MASK),
                              memory_order_release);
    }
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.mtx);
    for (int i = 0; i < P.nworkers; i++)
        pthread_join(P.threads[i], NULL);
    free(P.threads);
    free(P.wargs);
    P.threads = NULL;
    P.wargs = NULL;
    P.nworkers = 0;
    P.stop = 0;
}

void qwen_threadpool_after_fork(void) {
    P.threads = NULL;
    P.wargs = NULL;
    P.nworkers = 0;
    P.sleeping = 0;
    P.sleep_mask = 0;
    P.sleeping_hi = 0;
    P.main_sleeping = 0;
    atomic_store(&P.generation, 0);
    atomic_store(&P.completed, 0);
    atomic_store(&P.stop, 0);
    g_inited = 0;
}

void qwen_threadpool_start(int n_threads) {
    int want = n_threads > 1 ? n_threads - 1 : 0;
    if (!g_inited) {
        pthread_mutex_init(&P.submit_mtx, NULL);
        pthread_mutex_init(&P.mtx, NULL);
        pthread_cond_init(&P.wake, NULL);
        pthread_cond_init(&P.complete, NULL);
        atomic_store(&P.generation, 0);
        atomic_store(&P.completed, 0);
        P.sleeping = 0;
        P.sleep_mask = 0;
        P.sleeping_hi = 0;
        P.main_sleeping = 0;
        g_inited = 1;
    }
    if (want == P.nworkers) return;
    qwen_threadpool_stop();
    if (want == 0) return;
    P.threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)want);
    P.wargs   = (qwen_worker_arg_t *)malloc(sizeof(qwen_worker_arg_t) * (size_t)want);
    if (!P.threads || !P.wargs) {
        free(P.threads); free(P.wargs);
        P.threads = NULL; P.wargs = NULL; P.nworkers = 0; return;
    }
    int created = 0;
    unsigned long gen0 = atomic_load_explicit(&P.generation, memory_order_acquire);
    for (int i = 0; i < want; i++) {
        P.wargs[i].seen0 = gen0;
        P.wargs[i].idx   = i;
        if (pthread_create(&P.threads[i], NULL, worker_main, &P.wargs[i]) != 0) break;
        created++;
    }
    P.nworkers = created;
    if (created == 0) { free(P.threads); free(P.wargs); P.threads = NULL; P.wargs = NULL; }
}

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx) {
    if (nt == 0) return;
    if (!g_inited || P.nworkers == 0 || nt == 1) {
        PS_INC(ps_serial);
        for (size_t i = 0; i < nt; i++) fn(i, nt, ctx);
        return;
    }
    PS_INC(ps_dispatch);
    if (g_qp_meter) atomic_fetch_add(&g_qp_dispatches, 1);
    qwen_region_begin(QWEN_RGN_RT_POOL_DISPATCH);
    qwen_job_t job;
    job.fn = fn; job.ctx = ctx; job.nt = nt; job.tag = g_qwen_tls_tag;
    atomic_init(&job.next, 0);

    qwen_region_begin(QWEN_RGN_RT_POOL_SUBMIT);
    if (g_qp_low_until > 0.0) {
        /* LOW: step aside while the frame loop is in a burst of dispatches; only take the
         * pool in a window the loop is not using.  Past the deadline this is skipped. */
        double win = qwen_pool_hi_window_ms();
        for (;;) {
            double now = qwen_parallel_now_ms();
            if (now >= g_qp_low_until) { g_qp_low_until = 0.0; break; }
            if (now - atomic_load_explicit(&g_qp_last_hi_ms, memory_order_relaxed) >= win) break;
            qwen_cpu_relax();
        }
    }
    pthread_mutex_lock(&P.submit_mtx);
    if (g_qp_low_until <= 0.0)
        atomic_store_explicit(&g_qp_last_hi_ms, qwen_parallel_now_ms(), memory_order_relaxed);
    qwen_region_end(QWEN_RGN_RT_POOL_SUBMIT);

    int need = (int)nt - 1;
    if (need > P.nworkers) need = P.nworkers;
    if (need < 0) need = 0;
    if (!qwen_pool_narrow()) need = P.nworkers;

    P.job = &job;
    atomic_store_explicit(&P.completed, 0, memory_order_relaxed);
    pthread_mutex_lock(&P.mtx);
    {
        unsigned long gw = atomic_load_explicit(&P.generation, memory_order_relaxed);
        atomic_store_explicit(&P.generation,
                              QWEN_GW_MAKE(QWEN_GW_GEN(gw) + 1, need),
                              memory_order_release);
    }
    if (P.sleeping > 0) {
        const unsigned long long need_mask =
            need >= 64 ? ~0ULL : ((1ULL << need) - 1ULL);
        if ((P.sleep_mask & need_mask) || (P.sleeping_hi > 0 && need > 64))
            pthread_cond_broadcast(&P.wake);
    }
    pthread_mutex_unlock(&P.mtx);

    run_chunks(&job);

    /* The caller has finished its own chunks: everything from here to `completed ==
     * need` is time spent WAITING FOR THE WORKERS, and nothing else.  That is the one
     * kind of sync wait this pool can attribute honestly. */
    qwen_region_begin(QWEN_RGN_RT_POOL_WAIT);
    int budget = qwen_pool_spin();
    while (budget-- > 0 &&
           atomic_load_explicit(&P.completed, memory_order_acquire) != need)
        qwen_cpu_relax();
    if (atomic_load_explicit(&P.completed, memory_order_acquire) != need) {
        pthread_mutex_lock(&P.mtx);
        PS_INC(ps_main_park);
        P.main_sleeping = 1;
        while (atomic_load_explicit(&P.completed, memory_order_relaxed) != need)
            pthread_cond_wait(&P.complete, &P.mtx);
        P.main_sleeping = 0;
        pthread_mutex_unlock(&P.mtx);
    }
    qwen_region_end(QWEN_RGN_RT_POOL_WAIT);
    P.job = NULL;
    pthread_mutex_unlock(&P.submit_mtx);
    qwen_region_end(QWEN_RGN_RT_POOL_DISPATCH);
}

int qwen_parallel_team(void) {
    return (g_inited && P.nworkers > 0) ? P.nworkers + 1 : 1;
}

/* A worker inside run_chunks() cannot take submit_mtx: the outer submitter holds it until
 * every chunk has completed, so a nested dispatch deadlocks.  Two independent threads are
 * fine -- submit_mtx serialises them, the second simply waits for the first. */
int qwen_pool_nested_dispatch_ok(void) { return 0; }
int qwen_pool_concurrent_submit_ok(void) { return 1; }
int qwen_pool_priority_ok(void) { return 1; }

#endif

#if defined(__APPLE__) && defined(__BLOCKS__) && !defined(QWEN_FORCE_PTHREAD)
void qwen_pool_stats_report(void) { }
/* GCD has no spin/narrow knobs: report the values as "not applicable". */
int qwen_pool_spin_value(void)   { return -1; }
int qwen_pool_narrow_value(void) { return -1; }
#endif
