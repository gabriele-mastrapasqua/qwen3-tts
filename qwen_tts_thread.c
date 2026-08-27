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

/* The per-thread tag, and its propagation into the pool. Kept HERE rather than in
 * the kernels so that every backend's qwen_parallel (GCD, Win32, pthreads) carries it
 * with exactly one line, and no kernel has to know how threads are made. */
static __thread int g_qwen_tls_tag = 0;
int  qwen_tls_tag_get(void) { return g_qwen_tls_tag; }
void qwen_tls_tag_set(int tag) { g_qwen_tls_tag = tag; }
#include "qwen_tts_kernels.h"   /* qwen_ftz_on() */

/* -------------------------------------------------------------------------
 * macOS / GCD
 * ------------------------------------------------------------------------- */
#if defined(__APPLE__) && defined(__BLOCKS__) && !defined(QWEN_FORCE_PTHREAD)

#include <dispatch/dispatch.h>

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx) {
    if (nt == 0) return;
    if (nt == 1) { fn(0, 1, ctx); return; }
    const int tag = g_qwen_tls_tag;
    dispatch_apply(nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t tid) {
        qwen_ftz_on();              /* per GCD worker: flush denormals */
        qwen_tls_tag_set(tag);      /* adopt the submitter's audit tag */
        fn(tid, nt, ctx);
    });
}

void qwen_threadpool_start(int n_threads) { (void)n_threads; }
void qwen_threadpool_stop(void) {}
void qwen_threadpool_after_fork(void) {}   /* GCD owns no pool of ours */
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
    int tag;                /* submitter's audit tag, adopted by each worker */
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
        qwen_tls_tag_set(job->tag);   /* adopt the submitter's audit tag */
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

/* Windows has no fork(); the pre-fork server is Linux-only. Present so the symbol
 * resolves on every backend. */
void qwen_threadpool_after_fork(void) {}

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
    job.fn = fn; job.ctx = ctx; job.nt = nt; job.next = 0; job.tag = g_qwen_tls_tag;
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
    int tag;                /* submitter's audit tag, adopted by each worker */
} qwen_job_t;

/* Per-worker startup argument. It used to be just the initial `seen` generation cast
 * into a void*; the narrow-barrier work needs a STABLE INDEX per worker too, and
 * packing two things into a pointer is how you get a bug you cannot read. */
typedef struct { unsigned long seen0; int idx; } qwen_worker_arg_t;

/* The packed generation word (see P.generation). 16 bits of `needed` is 65535
 * workers; the generation half still has 48 bits, i.e. it cannot wrap in any run
 * this engine will ever have. */
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
    /* ── ONE word carries both "there is a new job" and "how wide is it" ──────────
     * genword = (generation << QWEN_GW_NEED_BITS) | needed. Packed, and not two
     * fields, for two reasons that are both measured:
     *   - a worker's wait predicate is "new generation AND it wants me", and split
     *     across two atomics that is two acquire loads per dispatch on lines the
     *     whole pool contends for (+12.5% on the dispatch cost, 2026-08-24);
     *   - packed, the two halves cannot be observed out of step, so the ordering
     *     argument for `needed` (see the long note that used to live here) reduces
     *     to the one the generation counter already had.
     * `needed` counts WORKERS only - main is extra and never counted. It must stay
     * readable by the last worker AFTER its fetch_add on `completed`, when main may
     * already have cleared P.job and returned, so it cannot live in the job struct
     * (that is main's stack frame). submit_mtx serialises submissions, so the word is
     * stable for the whole job. */
    _Atomic unsigned long generation;
    _Atomic int completed;     /* workers that finished the current job; main spins on it */
    int sleeping;              /* workers parked on `wake` — guarded by mtx */
    /* Which workers are parked, by index — guarded by mtx, like `sleeping`.
     * The dispatcher broadcasts only when a worker the job actually NEEDS is asleep;
     * without this, every narrow dispatch wakes the idle tail just to have it re-check
     * the predicate and park again, which is the futex storm the spin budget exists to
     * avoid. Indices >= 64 do not fit the mask and are counted separately (a pool that
     * big has never been run; the fallback is "broadcast", i.e. always correct). */
    unsigned long long sleep_mask;
    int sleeping_hi;
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
/* ── Il valore, e perche' e' diverso per piattaforma (2026-08-21) ──────────────
 * 4096 era tarato altrove e su ARM Linux costa il 40% del Code Predictor. Misurato sul
 * GCP c4a-standard-16 (Axion, Neoverse-V2), 1.7B --int8, richiesta singola:
 *
 *   -j8   SPIN=4096   CP 16,0 ms/f · RTF 0,45 · 491 320 context switch
 *   -j8   SPIN=16384  CP 11,7      · RTF 0,39 · 193 387
 *   -j8   SPIN=65536  CP  9,6      · RTF 0,35 ·  35 132   <- scelto
 *   -j8   SPIN=262144 CP  9,6      · RTF 0,35 ·   5 977   (nessun guadagno in piu')
 *
 * A livello server (-j16, c=4) porta Q da 1,94 a 2,43 e il p95 da 720 a 692 ms, e fa
 * tornare -j16 migliore di -j8: il tetto "un server non usa 16 core" era QUESTO.
 *
 * ⚠️ Perche' non si alza ovunque. Filare costa CPU mentre i worker aspettano: e' gratis
 * se quel core sarebbe rimasto inattivo, ed e' una tassa se serviva a qualcun altro.
 * Misurato che NON danneggia la topologia a piu' processi a carico basso (4 server:
 * p95 448 contro 439 ms), ma sotto saturazione con piu' processi non e' stato misurato.
 * E 262144 su -j16 PEGGIORA (31,4 ms/f contro 22,8): oltre un certo punto sedici thread
 * che filano si mangiano i core che servono al lavoro rimasto.
 *
 * x86 Linux paga lo stesso futex per dispatch e quasi certamente vuole lo stesso valore,
 * ma qui NON e' stato misurato: resta a 4096 finche' non gira su una scatola x86.
 * QWEN_POOL_SPIN scavalca sempre; 0 = parcheggia subito. */
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
/* ── Narrow barrier gate — still OPT-IN, but for ONE reason now, not three ───────────
 *
 * What it does: main waits for `nt-1` workers instead of all `nworkers`, and workers
 * beyond that index touch neither `job->next` nor `P.completed`.
 *
 * Three things kept it off by default on 2026-08-24. Two of them are GONE, killed by
 * the packed generation word + selective broadcast above (same day, TODO-1):
 *
 *  1. ~~not free as a no-op~~ FIXED. It used to cost one extra acquire load of P.needed
 *     per worker per job (+12.5% on the dispatch). `needed` now rides in the generation
 *     word, so the no-op case reads ONE atomic. Control cell, pool 8: 2023 -> 1885 ns.
 *     End-to-end the penalty is not visible in any topology measured (-j8, -j16, 2x8).
 *  2. ~~does not deliver what it was for~~ FIXED. A pool of 16 narrowed to nt=8 used to
 *     cost +39% CPU over a REAL pool of 8 because the idle tail kept spinning. It now
 *     costs the same: 1856 ns / 765 ms against 1991 ns / 768 ms on the bench, and
 *     5.46 against 5.42 average cores on the server.
 *  3. IT STILL HAS NO CONSUMER THAT BENEFITS. Every kernel call site passes
 *     nt = g_n_threads, so the gate does not bite. The one caller that narrows it,
 *     qwen_set_threads_soft() behind QWEN_THREADS_TALKER, IS wired into the batched
 *     server - and measured there it is a WORSE configuration than not narrowing at
 *     all: `-j16` + talker 8 gives TTFA p95 522 ms and RTF 1.45 at C=4, against 383 ms
 *     and 1.20 for plain `-j16`. Narrowing the Talker on a dedicated 16-core box frees
 *     cores that had no other claimant.
 *
 * So: a default changes when there is evidence it is BETTER, not merely evidence it is
 * not worse - and there is none in any shipping configuration. It becomes interesting
 * where the freed cores have a beneficiary (multi-process topologies, co-tenancy), which
 * is not measured. QWEN_POOL_NARROW=1 turns it on.
 * Numbers: the design notes */
static int qwen_pool_narrow(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("QWEN_POOL_NARROW");
        v = (e && e[0] == '1');
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

/* ── QWEN_POOL_STATS: dispatch accounting, compiled out by default ────────────────
 * Off unless the TU is built with -DQWEN_POOL_STATS, because the counters live on
 * lines the pool already contends for and measuring must not move what it measures.
 * Built in, they answer the only question the microbenchmark cannot: how many
 * qwen_parallel dispatches a frame actually costs. Report goes to stderr at exit. */
#ifdef QWEN_POOL_STATS
#include <stdio.h>
static _Atomic unsigned long ps_dispatch, ps_chunks, ps_worker_park, ps_main_park, ps_serial;
#define PS_INC(c) atomic_fetch_add_explicit(&(c), 1, memory_order_relaxed)
/* Reported from a destructor AND from atexit. The destructor alone was silent on the
 * batched server (2026-08-24): the shape census, which registers with atexit, printed
 * from the same run - so the two hooks are not equivalent here and a counter that only
 * has one of them reads as "zero dispatches" when it means "never printed". */
static void ps_report_body(void);
static void ps_report(void);
__attribute__((constructor)) static void ps_register(void) { atexit(ps_report); }
__attribute__((destructor)) static void ps_report_dtor(void) { ps_report(); }
static void ps_report(void) {
    static _Atomic int done = 0;
    if (atomic_exchange_explicit(&done, 1, memory_order_relaxed)) return;
    ps_report_body();
}
/* Callable explicitly, because a prefork WORKER ends with _exit(): neither atexit nor a
 * destructor runs there, so on 2026-08-24 a whole instrumented pass reported the PARENT's
 * counters (dispatch=6 for an entire campaign) and measured nothing. */
void qwen_pool_stats_report(void) { ps_report(); }
static void ps_report_body(void) {
    fprintf(stderr, "POOLSTATS dispatch=%lu chunks=%lu worker_park=%lu main_park=%lu serial=%lu\n",
            ps_dispatch, ps_chunks, ps_worker_park, ps_main_park, ps_serial);
}
#else
void qwen_pool_stats_report(void) { }   /* not built with QWEN_POOL_STATS */
#define PS_INC(c) ((void)0)
#endif

static void run_chunks(qwen_job_t *job) {
    size_t i;
    qwen_tls_tag_set(job->tag);   /* once per job, not per chunk */
    while ((i = atomic_fetch_add(&job->next, 1)) < job->nt)
    {   PS_INC(ps_chunks);
        job->fn(i, job->nt, job->ctx); }
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
    const qwen_worker_arg_t *wa = (const qwen_worker_arg_t *)arg;
    const int my_idx = wa->idx;
    const unsigned long long my_bit = my_idx < 64 ? (1ULL << my_idx) : 0ULL;
    unsigned long seen = wa->seen0;
    for (;;) {
        /* ── The wait predicate: a NEW generation that WANTS ME ──────────────────
         * Both halves come out of one acquire load, so a worker never spins on a
         * generation it has no work in. That is what makes an over-wide pool cheap:
         * the tail beyond `needed` parks ONCE and is then never woken again (the
         * dispatcher below broadcasts only when a needed worker is asleep), instead
         * of burning a full spin budget per dispatch. Measured 2026-08-24: without
         * this, a pool of 16 narrowed to nt=8 still cost +39% CPU over a real pool
         * of 8 - the barrier was fixed but the idle tail kept spinning. */
        unsigned long gw = atomic_load_explicit(&P.generation, memory_order_acquire);
        int budget = qwen_pool_spin();
        while (budget-- > 0 && !P.stop &&
               !(gw != seen && QWEN_GW_NEED(gw) > my_idx)) {
            qwen_cpu_relax();
            gw = atomic_load_explicit(&P.generation, memory_order_acquire);
        }

        if (!P.stop && !(gw != seen && QWEN_GW_NEED(gw) > my_idx)) {
            /* Nothing for me: park. Re-check the predicate under the lock (see note). */
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

        /* Re-load with acquire: the park path's last read was relaxed, and this is the
         * edge that publishes P.job and the job's contents to this worker. */
        seen = atomic_load_explicit(&P.generation, memory_order_acquire);
        const int need = QWEN_GW_NEED(seen);
        qwen_job_t *job = P.job;
        if (job && my_idx < need) {
            run_chunks(job);
            /* Completion: publish lock-free (main spins on it); only pay the futex
             * to wake main if it has actually parked. */
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
    /* Bump generation too: a worker spinning (not parked) must fall through its
     * spin and observe stop, not spin forever waiting for a job that won't come.
     * `needed` goes to the maximum so the bump satisfies every worker's predicate as
     * well - P.stop alone would do, but a shutdown is not the place to depend on one
     * of two exit conditions. */
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

/* After fork: the worker threads are gone, the struct is not. Reinitialise rather
 * than reuse - a mutex held by a thread that no longer exists is undefined, and the
 * pthread_t array points at ids that will never be joined. The memory is the child's
 * own copy-on-write page, so freeing it here costs one page and leaks nothing. */
void qwen_threadpool_after_fork(void) {
    P.threads = NULL;      /* deliberately not free()d: the parent's allocator state
                            * is inherited mid-flight and free() here has burned us
                            * before. One pointer of leak per fork, once. */
    P.wargs = NULL;        /* same reasoning */
    P.nworkers = 0;
    P.sleeping = 0;
    P.sleep_mask = 0;
    P.sleeping_hi = 0;
    P.main_sleeping = 0;
    atomic_store(&P.generation, 0);
    atomic_store(&P.completed, 0);
    atomic_store(&P.stop, 0);
    g_inited = 0;          /* forces a full re-init on the next start() */
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
    if (!P.threads || !P.wargs) {                /* audit #9: fall back to serial */
        free(P.threads); free(P.wargs);
        P.threads = NULL; P.wargs = NULL; P.nworkers = 0; return;
    }
    /* audit #9: cap nworkers to threads actually created; qwen_parallel runs
     * serially when nworkers==0 and correctly with a partial pool otherwise. */
    int created = 0;
    /* Hand each worker the CURRENT generation as its initial `seen`: any bump after
     * this point (first possible submit is after start() returns) is then observable. */
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
    qwen_job_t job;
    job.fn = fn; job.ctx = ctx; job.nt = nt; job.tag = g_qwen_tls_tag;
    atomic_init(&job.next, 0);

    /* Serialize the whole submit→run→wait against the single job slot (see submit_mtx
     * comment): two concurrent submitters would otherwise corrupt job/completed/generation. */
    pthread_mutex_lock(&P.submit_mtx);

    /* ── How wide is this job, really? ──────────────────────────────────────────
     * `nt` is the number of chunks the caller wants, and MAIN takes one of them, so
     * the job needs nt-1 workers. It used to need all of them: main waited for
     * `completed == P.nworkers` regardless of nt, so a job asking for 8 chunks on a
     * 16-thread pool still paid a 15-worker barrier plus 15 useless fetch_adds on
     * job->next. That is exactly why qwen_set_threads_soft() measured as a no-op on
     * 2026-08-24. QWEN_POOL_NARROW=0 restores the old behaviour without a rebuild. */
    int need = (int)nt - 1;
    if (need > P.nworkers) need = P.nworkers;
    if (need < 0) need = 0;
    if (!qwen_pool_narrow()) need = P.nworkers;

    P.job = &job;
    atomic_store_explicit(&P.completed, 0, memory_order_relaxed);
    /* Publish the job UNDER mtx so a worker that is about to park (and re-checks the
     * predicate under mtx) cannot miss it. `needed` rides in the same word as the
     * generation, so a worker that observes the bump observes the right width - one
     * release store, not two. */
    pthread_mutex_lock(&P.mtx);
    {
        unsigned long gw = atomic_load_explicit(&P.generation, memory_order_relaxed);
        atomic_store_explicit(&P.generation,
                              QWEN_GW_MAKE(QWEN_GW_GEN(gw) + 1, need),
                              memory_order_release);
    }
    /* Broadcast ONLY if a worker this job needs is actually parked. Waking the idle
     * tail so it can re-read the predicate and park again is a futex round-trip per
     * dispatch per idle worker - the exact cost the spin budget exists to avoid, and
     * the reason an over-wide pool used to be dearer than a right-sized one. Workers
     * that are spinning need no wake at all: they see the new word lock-free. */
    if (P.sleeping > 0) {
        const unsigned long long need_mask =
            need >= 64 ? ~0ULL : ((1ULL << need) - 1ULL);
        if ((P.sleep_mask & need_mask) || (P.sleeping_hi > 0 && need > 64))
            pthread_cond_broadcast(&P.wake);
    }
    pthread_mutex_unlock(&P.mtx);

    run_chunks(&job);              /* main participates */

    /* Wait for the workers: spin on the atomic count, then park. */
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
    P.job = NULL;
    pthread_mutex_unlock(&P.submit_mtx);
}

/* ── Reentranza: SÌ da un altro thread, NO annidata dentro un task ────────────────
 *
 * Diceva 0 "perche' c'e' un solo job slot globale". Il job slot e' uno, ma `submit_mtx`
 * serializza l'INTERO submit→run→wait: il secondo sottomettitore si blocca, non corrompe
 * niente. E quando il primo rilascia il mutex ha gia' fatto `P.job = NULL` DOPO aver
 * atteso `completed == nworkers`, quindi ogni worker ha finito il suo `run_chunks` — non
 * c'e' nessuna finestra in cui un worker legge il job del sottomettitore sbagliato. Un
 * worker lento che deve ancora tornare in cima al loop ha `seen` alla generazione vecchia,
 * vede quella nuova e prende il job nuovo: corretto.
 *
 * ⚠️ Quello che resta VIETATO e' la sottomissione ANNIDATA: `qwen_parallel` chiamata da
 * dentro una task function girerebbe su un worker che poi si blocca su `submit_mtx`,
 * mentre il main tiene il mutex aspettando `completed == nworkers` → deadlock. Era vero
 * anche prima; questo flag non lo copriva e non lo copre. La regola e': chiamare
 * qwen_parallel da un THREAD, mai da un TASK.
 *
 * Verificato il 2026-08-18 prima di cambiare il flag, non dato per scontato: estratte le
 * 28 funzioni `*_task` di qwen_tts_kernels.c col brace matching e cercata `qwen_parallel(`
 * nei loro corpi → zero occorrenze. Tutti i call site stanno nei DISPATCHER (qwen_matvec_*
 * / qwen_matmat_*), che girano sul thread chiamante. Se un giorno un task deve
 * parallelizzare al suo interno, questo flag va rimesso a 0 O il pool deve diventare una
 * coda: e' quello il vincolo, non "quanti thread chiamano".
 *
 * PERCHE' CAMBIARLO ORA (misurato il 2026-08-18 sul c3). `is_reentrant()` e' il gate di
 * A1, l'helper di prefill asincrono in qwen_tts.c:3148. Tornando 0, su Linux/Windows
 * l'helper non parte MAI e si prende il ramo `inline fallback (non-reentrant pool):
 * prefill blocks the batch`: quattro arrivi simultanei = quattro prefill in fila DENTRO il
 * frame loop, e il TTFA del quarto se li porta tutti. Sul c3 questo vale `admission +
 * prefill` 23-30% del loop e un degrado TTFA di 13.9x da c=1 a c=4. In pratica A1 era
 * un'ottimizzazione attiva solo su macOS, cioe' ovunque tranne in produzione.
 *
 * 🚨 E LA MISURA HA DETTO DI NO — torna 0, ma per il motivo GIUSTO.
 * Provato ad accenderlo (c3, 1.7B base, -j4, int8, REQS=16, testi misti, celle 'match',
 * A/B con lo stesso binario via QWEN_PREFILL_HELPER):
 *
 *     c    TTFA p95 helper ON    TTFA p95 helper OFF     Q ON / OFF
 *     1        3053 ms               3087 ms            0.59 / 0.59   <- controllo: uguale
 *     2        4338 ms               2625 ms            0.63 / 0.67
 *     4        7110 ms               5477 ms            0.78 / 0.81
 *
 * Accenderlo PEGGIORA il TTFA del ~30% e il throughput del ~4%. La cella c=1 e' identica
 * nei due bracci (li' l'helper non entra in gioco), quindi non e' deriva della macchina.
 *
 * Perche': con UN SOLO job slot le sezioni parallele dell'helper e quelle del frame loop
 * si serializzano su `submit_mtx` invece di sovrapporsi. Si aggiunge contesa sul mutex e
 * cambi di contesto su 4 core gia' saturi, e non si guadagna parallelismo — il prefill
 * esce dal percorso critico del frame loop solo per rientrarci dalla porta del lock.
 *
 * QUINDI: la reentranza NON e' il pezzo mancante; il pezzo mancante e' che il pool
 * diventi una CODA (piu' job in volo, worker che pescano il primo disponibile). Finche'
 * il pool ha un job slot, A1 su Linux e' una pessimizzazione e questo torna 0.
 * QWEN_PREFILL_HELPER=1 resta come braccio sperimentale: e' la manopola con cui
 * rimisurare il giorno in cui il pool cambia, senza rimettere mano a questo file. */
int qwen_parallel_is_reentrant(void) {
    const char *e = getenv("QWEN_PREFILL_HELPER");
    return (e && e[0] == '1') ? 1 : 0;
}

#endif
