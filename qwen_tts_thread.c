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
    qwen_ftz_on();
    EnterCriticalSection(&P.mtx);
    /* Initial `seen` = generation at create time (passed by the creator), not a read
     * done at first schedule: a submit racing thread startup would be missed → hang. */
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
    unsigned long gen0 = P.generation;  /* workers' initial `seen` (see worker_main) */
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
#include <stdint.h>
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
    _Atomic unsigned long generation; /* bumped per dispatch; spun on lock-free, so atomic */
    _Atomic int completed;     /* workers that finished the current job; main spins on it */
    int sleeping;              /* workers parked on `wake` — guarded by mtx */
    int main_sleeping;         /* main parked on `complete` — guarded by mtx */
    _Atomic int stop;          /* read in the worker spin loop outside the mutex → atomic */
} P;
static int g_inited = 0;

/* Spin budget: how many generation re-reads a worker (or the main thread's
 * completion wait) does before parking on the condvar. The POSIX pool used to
 * pay a futex round-trip per dispatch (~7300/frame, per PR #17). Spinning first
 * skips the syscall when the next job lands within a few µs — which it does at
 * every frame boundary. QWEN_POOL_SPIN overrides; 0 = never spin (park at once).
 * Lower it when synthesis overlaps other CPU-heavy work so idle spin does not
 * steal those cores. */
#define QWEN_POOL_SPIN_DEFAULT 4096
static int qwen_pool_spin(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_POOL_SPIN");
        v = e ? atoi(e) : QWEN_POOL_SPIN_DEFAULT;
        if (v < 0) v = 0;
    }
    return v;
}
static inline void qwen_cpu_relax(void) {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static void run_chunks(qwen_job_t *job) {
    size_t i;
    while ((i = atomic_fetch_add(&job->next, 1)) < job->nt)
        job->fn(i, job->nt, job->ctx);
}

/* Correctness note (lost-wakeup avoidance). generation is bumped by the
 * dispatcher UNDER P.mtx, and a worker's decision to park re-checks generation
 * UNDER P.mtx after incrementing `sleeping`. So the two orderings are:
 *   - worker parks first: dispatcher then sees sleeping>0 and broadcasts.
 *   - dispatcher bumps first: the worker's under-lock re-check sees the new
 *     generation and does NOT park.
 * There is no Dekker/store-buffer race because `sleeping` is only ever read and
 * written under the mutex; the lock-free path is ONLY the spin, which reads the
 * atomic generation and never decides to sleep. Same structure guards the
 * completion side with `main_sleeping`/`completed`. */
static void *worker_main(void *arg) {
    qwen_ftz_on();             /* per-thread FTZ (int8 denormals) — set once */
    /* Initial `seen` comes from the creator (generation at create time), NOT from a
     * load done when the OS first schedules this thread: a submit racing the thread
     * startup would otherwise be absorbed into `seen` and the job missed → deadlock
     * (submitter waits completed==nworkers forever; hit by --self-test/--matmat-bench,
     * which dispatch immediately after threadpool_start). */
    unsigned long seen = (unsigned long)(uintptr_t)arg;
    for (;;) {
        /* Spin on the atomic generation before paying the futex. */
        int budget = qwen_pool_spin();
        while (budget-- > 0 &&
               atomic_load_explicit(&P.generation, memory_order_acquire) == seen &&
               !P.stop)
            qwen_cpu_relax();

        if (atomic_load_explicit(&P.generation, memory_order_acquire) == seen) {
            /* Still nothing: park. Re-check generation under the lock (see note). */
            pthread_mutex_lock(&P.mtx);
            P.sleeping++;
            while (!P.stop &&
                   atomic_load_explicit(&P.generation, memory_order_relaxed) == seen)
                pthread_cond_wait(&P.wake, &P.mtx);
            P.sleeping--;
            pthread_mutex_unlock(&P.mtx);
        }
        if (P.stop) break;

        seen = atomic_load_explicit(&P.generation, memory_order_acquire);
        qwen_job_t *job = P.job;
        if (job) run_chunks(job);

        /* Completion: publish lock-free (main spins on it); only pay the futex
         * to wake main if it has actually parked. */
        if (atomic_fetch_add_explicit(&P.completed, 1, memory_order_acq_rel) + 1
                == P.nworkers) {
            pthread_mutex_lock(&P.mtx);
            if (P.main_sleeping) pthread_cond_signal(&P.complete);
            pthread_mutex_unlock(&P.mtx);
        }
    }
    return NULL;
}

void qwen_threadpool_stop(void) {
    if (!g_inited || !P.threads) return;
    pthread_mutex_lock(&P.mtx);
    P.stop = 1;
    /* Bump generation too: a worker spinning (not parked) must fall through its
     * spin and observe stop, not spin forever waiting for a job that won't come. */
    atomic_fetch_add_explicit(&P.generation, 1, memory_order_release);
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
        atomic_store(&P.generation, 0);
        atomic_store(&P.completed, 0);
        P.sleeping = 0;
        P.main_sleeping = 0;
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
    /* Hand each worker the CURRENT generation as its initial `seen`: any bump after
     * this point (first possible submit is after start() returns) is then observable. */
    unsigned long gen0 = atomic_load_explicit(&P.generation, memory_order_acquire);
    for (int i = 0; i < want; i++) {
        if (pthread_create(&P.threads[i], NULL, worker_main,
                           (void *)(uintptr_t)gen0) != 0) break;
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

    P.job = &job;
    atomic_store_explicit(&P.completed, 0, memory_order_relaxed);
    /* Publish the job UNDER mtx so a worker that is about to park (and re-checks
     * generation under mtx) cannot miss it. Only broadcast if someone is parked;
     * spinning workers pick the new generation up lock-free. */
    pthread_mutex_lock(&P.mtx);
    atomic_fetch_add_explicit(&P.generation, 1, memory_order_release);
    if (P.sleeping > 0) pthread_cond_broadcast(&P.wake);
    pthread_mutex_unlock(&P.mtx);

    run_chunks(&job);              /* main participates */

    /* Wait for the workers: spin on the atomic count, then park. */
    int budget = qwen_pool_spin();
    while (budget-- > 0 &&
           atomic_load_explicit(&P.completed, memory_order_acquire) != P.nworkers)
        qwen_cpu_relax();
    if (atomic_load_explicit(&P.completed, memory_order_acquire) != P.nworkers) {
        pthread_mutex_lock(&P.mtx);
        P.main_sleeping = 1;
        while (atomic_load_explicit(&P.completed, memory_order_relaxed) != P.nworkers)
            pthread_cond_wait(&P.complete, &P.mtx);
        P.main_sleeping = 0;
        pthread_mutex_unlock(&P.mtx);
    }
    P.job = NULL;
    pthread_mutex_unlock(&P.submit_mtx);
}

/* Single global job slot → NOT safe to submit from two threads at once. */
int qwen_parallel_is_reentrant(void) { return 0; }

#endif
