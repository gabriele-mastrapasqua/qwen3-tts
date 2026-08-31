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

#ifdef __cplusplus
}
#endif

#endif
