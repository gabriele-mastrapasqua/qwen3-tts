/*
 * qwen_tts_thread.c - Cross-OS parallel-for (PLAN 21.2)
 *
 * Backend selection:
 *   __APPLE__ + __BLOCKS__         -> GCD dispatch_apply
 *   _WIN32 (and not QWEN_USE_PTHREADS) -> Win32 threads + condition variables
 *   else                           -> pthread persistent pool (Linux, WSL, BSD)
 *
 * To exercise the pthread path on macOS for testing, build with
 * -DQWEN_FORCE_PTHREAD (overrides the GCD backend).
 */

#include "qwen_tts_thread.h"
#include "qwen_tts_kernels.h"   /* qwen_ftz_on() */

/* -------------------------------------------------------------------------
 * macOS / GCD
 * ------------------------------------------------------------------------- */
#if defined(__APPLE__) && defined(__BLOCKS__) && !defined(QWEN_FORCE_PTHREAD)

#include <dispatch/dispatch.h>

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx) {
    if (nt == 0) return;
    if (nt == 1) { fn(0, 1, ctx); return; }
    dispatch_apply(nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t tid) {
        qwen_ftz_on();              /* per GCD worker: flush denormals */
        fn(tid, nt, ctx);
    });
}

void qwen_threadpool_start(int n_threads) { (void)n_threads; }
void qwen_threadpool_stop(void) {}
int qwen_parallel_is_reentrant(void) { return 1; }  /* GCD: concurrent callers safe */

/* -------------------------------------------------------------------------
 * Windows native (Win32 threads + condition variables)
 * ------------------------------------------------------------------------- */
#elif defined(_WIN32) && !defined(QWEN_USE_PTHREADS)

#include <windows.h>

typedef struct {
    qwen_task_fn fn;
    void *ctx;
    size_t nt;
    volatile LONG64 next;   /* next chunk to claim (InterlockedIncrement64) */
} qwen_job_t;

static struct {
    HANDLE *threads;
    int nworkers;
    CRITICAL_SECTION mtx;
    CONDITION_VARIABLE wake;     /* workers wait for a new job */
    CONDITION_VARIABLE complete; /* main waits for job completion */
    qwen_job_t *job;
    unsigned long generation;
    int completed;
    int stop;
} P;
static int g_inited = 0;

static void run_chunks(qwen_job_t *job) {
    /* InterlockedIncrement64 returns the post-increment value, so claim i-1. */
    LONG64 v;
    while ((v = InterlockedIncrement64(&job->next)) <= (LONG64)job->nt) {
        job->fn((size_t)(v - 1), job->nt, job->ctx);
    }
}

static DWORD WINAPI worker_main(LPVOID arg) {
    (void)arg;
    qwen_ftz_on();
    EnterCriticalSection(&P.mtx);
    unsigned long seen = P.generation;
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

void qwen_threadpool_start(int n_threads) {
    int want = n_threads > 1 ? n_threads - 1 : 0;  /* main participates */
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
    if (!P.threads) { P.nworkers = 0; return; }  /* audit #9: fall back to serial */
    int created = 0;
    for (int i = 0; i < want; i++) {
        P.threads[i] = CreateThread(NULL, 0, worker_main, NULL, 0, NULL);
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
    job.fn = fn; job.ctx = ctx; job.nt = nt; job.next = 0;
    EnterCriticalSection(&P.mtx);
    P.job = &job;
    P.completed = 0;
    P.generation++;
    WakeAllConditionVariable(&P.wake);
    LeaveCriticalSection(&P.mtx);

    run_chunks(&job);              /* main participates */

    EnterCriticalSection(&P.mtx);
    while (P.completed != P.nworkers)
        SleepConditionVariableCS(&P.complete, &P.mtx, INFINITE);
    P.job = NULL;
    LeaveCriticalSection(&P.mtx);
}

/* Single global job slot → NOT safe to submit from two threads at once. */
int qwen_parallel_is_reentrant(void) { return 0; }

/* -------------------------------------------------------------------------
 * POSIX pthread persistent pool (Linux / WSL / *BSD; macOS with -DQWEN_FORCE_PTHREAD)
 * ------------------------------------------------------------------------- */
#else

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef struct {
    qwen_task_fn fn;
    void *ctx;
    size_t nt;
    atomic_size_t next;     /* next chunk to claim (shared main + workers) */
} qwen_job_t;

static struct {
    pthread_t *threads;
    int nworkers;
    pthread_mutex_t submit_mtx; /* serialize the whole submit→run→wait cycle: the pool has a
                                 * SINGLE job slot, so two threads calling qwen_parallel at once
                                 * (e.g. batched-server scheduler + a reader/decode thread) would
                                 * clobber job/completed/generation → a worker could miss a job and
                                 * the submitter's cond_wait hangs (the intermittent 8.9s↔220s bug
                                 * seen on the EPYC batched server). This makes concurrent submitters
                                 * serialize instead — correct, since the pool runs one job at a time. */
    pthread_mutex_t mtx;
    pthread_cond_t wake;       /* workers wait for a new job */
    pthread_cond_t complete;   /* main waits for job completion */
    qwen_job_t *job;
    unsigned long generation;  /* bumped per dispatch; workers compare to detect new work */
    int completed;             /* workers that finished the current job */
    int stop;
} P;
static int g_inited = 0;

static void run_chunks(qwen_job_t *job) {
    size_t i;
    while ((i = atomic_fetch_add(&job->next, 1)) < job->nt)
        job->fn(i, job->nt, job->ctx);
}

static void *worker_main(void *arg) {
    (void)arg;
    qwen_ftz_on();             /* per-thread FTZ (int8 denormals) — set once */
    pthread_mutex_lock(&P.mtx);
    unsigned long seen = P.generation;
    for (;;) {
        while (!P.stop && P.generation == seen)
            pthread_cond_wait(&P.wake, &P.mtx);
        if (P.stop) break;
        seen = P.generation;
        qwen_job_t *job = P.job;
        pthread_mutex_unlock(&P.mtx);
        if (job) run_chunks(job);
        pthread_mutex_lock(&P.mtx);
        if (++P.completed == P.nworkers)
            pthread_cond_signal(&P.complete);
    }
    pthread_mutex_unlock(&P.mtx);
    return NULL;
}

void qwen_threadpool_stop(void) {
    if (!g_inited || !P.threads) return;
    pthread_mutex_lock(&P.mtx);
    P.stop = 1;
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.mtx);
    for (int i = 0; i < P.nworkers; i++)
        pthread_join(P.threads[i], NULL);
    free(P.threads);
    P.threads = NULL;
    P.nworkers = 0;
    P.stop = 0;
}

void qwen_threadpool_start(int n_threads) {
    int want = n_threads > 1 ? n_threads - 1 : 0;  /* main participates */
    if (!g_inited) {
        pthread_mutex_init(&P.submit_mtx, NULL);
        pthread_mutex_init(&P.mtx, NULL);
        pthread_cond_init(&P.wake, NULL);
        pthread_cond_init(&P.complete, NULL);
        P.generation = 0;
        g_inited = 1;
    }
    if (want == P.nworkers) return;
    qwen_threadpool_stop();
    if (want == 0) return;
    P.threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)want);
    if (!P.threads) { P.nworkers = 0; return; }  /* audit #9: fall back to serial */
    /* audit #9: cap nworkers to threads actually created; qwen_parallel runs
     * serially when nworkers==0 and correctly with a partial pool otherwise. */
    int created = 0;
    for (int i = 0; i < want; i++) {
        if (pthread_create(&P.threads[i], NULL, worker_main, NULL) != 0) break;
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
    job.fn = fn; job.ctx = ctx; job.nt = nt;
    atomic_init(&job.next, 0);

    /* Serialize the whole submit→run→wait against the single job slot (see submit_mtx
     * comment): two concurrent submitters would otherwise corrupt job/completed/generation. */
    pthread_mutex_lock(&P.submit_mtx);
    pthread_mutex_lock(&P.mtx);
    P.job = &job;
    P.completed = 0;
    P.generation++;
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.mtx);

    run_chunks(&job);              /* main participates */

    pthread_mutex_lock(&P.mtx);
    while (P.completed != P.nworkers)
        pthread_cond_wait(&P.complete, &P.mtx);
    P.job = NULL;
    pthread_mutex_unlock(&P.mtx);
    pthread_mutex_unlock(&P.submit_mtx);
}

/* Single global job slot → NOT safe to submit from two threads at once. */
int qwen_parallel_is_reentrant(void) { return 0; }

#endif
