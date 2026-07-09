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
    P.nworkers = want;
    for (int i = 0; i < want; i++)
        P.threads[i] = CreateThread(NULL, 0, worker_main, NULL, 0, NULL);
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

/* Hybrid spin-then-sleep pool. During the token loop, qwen_parallel dispatches
 * arrive back-to-back (hundreds per frame), and pure condvar handoff costs a
 * futex round-trip per dispatch — measured ~7300 futex calls per generated
 * frame on a 4-core Neoverse-N1. Workers therefore spin briefly for the next
 * job before sleeping, and the dispatcher only touches the condvar when a
 * worker is actually asleep. Chunk claiming stays dynamic, so outputs are
 * bit-identical to the sleeping pool. */

#define QWEN_POOL_SPIN_DEFAULT 8192 /* yield iterations (~40-80us) before sleeping */
/* Spin budget before a worker parks on the condvar. Overridable with
 * QWEN_POOL_SPIN: lower it when synthesis overlaps other CPU-heavy work
 * (e.g. the streaming decoder) so idle spin does not steal those cores. */
static int qwen_pool_spin(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_POOL_SPIN");
        v = e ? atoi(e) : QWEN_POOL_SPIN_DEFAULT;
        if (v < 1) v = 1;
    }
    return v;
}

typedef struct {
    qwen_task_fn fn;
    void *ctx;
    size_t nt;
    atomic_size_t next;     /* next chunk to claim (shared main + workers) */
} qwen_job_t;

static struct {
    pthread_t *threads;
    int nworkers;
    pthread_mutex_t mtx;
    pthread_cond_t wake;       /* workers wait for a new job (sleep phase only) */
    pthread_cond_t complete;   /* main waits for job completion (sleep phase only) */
    qwen_job_t *job;
    _Atomic unsigned long generation;  /* bumped per dispatch (release) */
    _Atomic int completed;     /* workers that finished the current job */
    _Atomic int sleeping;      /* workers inside the condvar sleep phase */
    _Atomic int main_sleeping; /* main is sleeping in the completion wait */
    _Atomic int stop;
} P;
static int g_inited = 0;

static inline void qwen_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#endif
}

static void run_chunks(qwen_job_t *job) {
    size_t i;
    while ((i = atomic_fetch_add(&job->next, 1)) < job->nt)
        job->fn(i, job->nt, job->ctx);
}

static void *worker_main(void *arg) {
    (void)arg;
    qwen_ftz_on();             /* per-thread FTZ (int8 denormals) — set once */
    unsigned long seen = atomic_load(&P.generation);
    for (;;) {
        /* Spin phase: catch back-to-back dispatches without a futex trip. */
        int spins = 0;
        while (!atomic_load_explicit(&P.stop, memory_order_relaxed) &&
               atomic_load_explicit(&P.generation, memory_order_acquire) == seen) {
            if (++spins >= qwen_pool_spin()) {
                /* Sleep phase. Re-check generation under the mutex before
                 * waiting so a dispatch between our last spin check and the
                 * cond_wait cannot be missed. */
                pthread_mutex_lock(&P.mtx);
                atomic_fetch_add(&P.sleeping, 1);
                while (!atomic_load(&P.stop) && atomic_load(&P.generation) == seen)
                    pthread_cond_wait(&P.wake, &P.mtx);
                atomic_fetch_sub(&P.sleeping, 1);
                pthread_mutex_unlock(&P.mtx);
                break;
            }
            qwen_cpu_relax();
        }
        if (atomic_load(&P.stop)) break;
        seen = atomic_load_explicit(&P.generation, memory_order_acquire);
        qwen_job_t *job = P.job;   /* published before the generation bump */
        if (job) run_chunks(job);
        if (atomic_fetch_add(&P.completed, 1) + 1 == P.nworkers &&
            atomic_load(&P.main_sleeping)) {
            pthread_mutex_lock(&P.mtx);
            pthread_cond_signal(&P.complete);
            pthread_mutex_unlock(&P.mtx);
        }
    }
    return NULL;
}

void qwen_threadpool_stop(void) {
    if (!g_inited || !P.threads) return;
    pthread_mutex_lock(&P.mtx);
    atomic_store(&P.stop, 1);
    pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.mtx);
    for (int i = 0; i < P.nworkers; i++)
        pthread_join(P.threads[i], NULL);
    free(P.threads);
    P.threads = NULL;
    P.nworkers = 0;
    atomic_store(&P.stop, 0);
}

void qwen_threadpool_start(int n_threads) {
    int want = n_threads > 1 ? n_threads - 1 : 0;  /* main participates */
    if (!g_inited) {
        pthread_mutex_init(&P.mtx, NULL);
        pthread_cond_init(&P.wake, NULL);
        pthread_cond_init(&P.complete, NULL);
        atomic_init(&P.generation, 0);
        atomic_init(&P.completed, 0);
        atomic_init(&P.sleeping, 0);
        atomic_init(&P.main_sleeping, 0);
        atomic_init(&P.stop, 0);
        g_inited = 1;
    }
    if (want == P.nworkers) return;
    qwen_threadpool_stop();
    if (want == 0) return;
    P.threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)want);
    P.nworkers = want;
    for (int i = 0; i < want; i++)
        pthread_create(&P.threads[i], NULL, worker_main, NULL);
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

    P.job = &job;
    atomic_store(&P.completed, 0);
    atomic_fetch_add_explicit(&P.generation, 1, memory_order_release);
    /* Only pay the futex wake if someone is actually asleep. A worker that was
     * about to sleep re-checks the generation under the mutex, so it cannot
     * miss this dispatch even if we skip the broadcast here. */
    if (atomic_load(&P.sleeping) > 0) {
        pthread_mutex_lock(&P.mtx);
        pthread_cond_broadcast(&P.wake);
        pthread_mutex_unlock(&P.mtx);
    }

    run_chunks(&job);              /* main participates */

    /* Completion: spin briefly, then sleep. */
    int spins = 0;
    while (atomic_load_explicit(&P.completed, memory_order_acquire) != P.nworkers) {
        if (++spins >= qwen_pool_spin()) {
            pthread_mutex_lock(&P.mtx);
            atomic_store(&P.main_sleeping, 1);
            while (atomic_load(&P.completed) != P.nworkers)
                pthread_cond_wait(&P.complete, &P.mtx);
            atomic_store(&P.main_sleeping, 0);
            pthread_mutex_unlock(&P.mtx);
            break;
        }
        qwen_cpu_relax();
    }
    P.job = NULL;
}

/* Single global job slot → NOT safe to submit from two threads at once. */
int qwen_parallel_is_reentrant(void) { return 0; }

#endif
