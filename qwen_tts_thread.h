/* qwen_tts_thread.h - Cross-OS parallel-for abstraction */
#ifndef QWEN_TTS_THREAD_H
#define QWEN_TTS_THREAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*qwen_task_fn)(size_t tid, size_t nt, void *ctx);

void qwen_parallel(size_t nt, qwen_task_fn fn, void *ctx);

void qwen_threadpool_after_fork(void);

void qwen_parallel_meter(int on);
void qwen_parallel_meter_read(double *busy_ms, long long *chunks, long long *dispatches);

int  qwen_tls_tag_get(void);
void qwen_tls_tag_set(int tag);

void qwen_threadpool_start(int n_threads);

void qwen_threadpool_stop(void);
void qwen_pool_stats_report(void);

/* Two capability questions the pool must answer honestly, because three callers used to
 * ask one predicate that meant something different on each backend (on pthread it read the
 * QWEN_PREFILL_HELPER opt-in, which is a feature flag and not a capability at all).
 *
 * nested_dispatch_ok: may a task ALREADY running on the pool call qwen_parallel again?
 * concurrent_submit_ok: may two independent threads submit to the pool at the same time? */
int qwen_pool_nested_dispatch_ok(void);
int qwen_pool_concurrent_submit_ok(void);
/* Does qwen_parallel_set_low_until() actually deprioritise a submitter on this backend, or
 * is it a no-op?  A knob that silently does nothing is worse than one that says so. */
int qwen_pool_priority_ok(void);
/* 1 while the calling thread is executing a chunk of a qwen_parallel region (worker or
 * caller).  A nested dispatch from there would deadlock on the pool's single job slot, so
 * code that may run in both contexts asks this and runs its work inline when set. */
int qwen_parallel_active(void);
/* Dispatch priority.  A thread whose deadline is in the future submits LOW: its dispatch
 * waits while a normal (HIGH) submitter has dispatched within the last window, so LOW work
 * only fills the pool's idle windows between the frame loop's own dispatches.  Past the
 * deadline the thread is ordinary again, which bounds how long LOW work can be starved.
 * until_ms is CLOCK_MONOTONIC milliseconds (qwen_parallel_now_ms); 0 = HIGH (default). */
void   qwen_parallel_set_low_until(double until_ms);
double qwen_parallel_now_ms(void);

/* Persistent parallel regions.  qwen_parallel_team() is the number of threads that take
 * part in a dispatch of exactly that many chunks (the caller plus every pool worker), so a
 * region dispatched with nt == team can keep the whole team inside one task and separate
 * dependent phases with qwen_barrier_wait instead of leaving and re-dispatching.  A team
 * of 1 means the pool cannot promise that and the caller must use plain dispatches. */
int qwen_parallel_team(void);
typedef struct { volatile int arrived; volatile int phase; int nt; } qwen_barrier_t;
void qwen_barrier_init(qwen_barrier_t *b, int nt);
void qwen_barrier_wait(qwen_barrier_t *b);

#ifdef __cplusplus
}
#endif

const char *qwen_pool_flag_inert(const char *flag);

#endif
