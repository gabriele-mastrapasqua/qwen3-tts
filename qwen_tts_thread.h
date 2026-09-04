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

int qwen_parallel_is_reentrant(void);
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

#ifdef __cplusplus
}
#endif

#endif
