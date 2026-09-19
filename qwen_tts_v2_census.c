/* qwen_tts_v2_census.c - low-overhead, aggregated v2 execution evidence. */
#include "qwen_tts_v2_census.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_batch.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define V2_MAX_ROWS 512

int qwen_batch_chunk_plan(int B, int max_chunk, int *parts, int cap) {
    if (B <= 0 || max_chunk <= 0 || !parts || cap <= 0) return 0;
    if (max_chunk > 16) max_chunk = 16;
    int n = 0;
    for (int left = B; left > 0; left -= max_chunk) {
        if (n >= cap) return 0;
        parts[n++] = left > max_chunk ? max_chunk : left;
    }
    return n;
}

typedef struct {
    int stage, capacity, runnable, beff, contiguous;
    int weight, operation, path, leaf, reason;
    atomic_llong count, macs;
} v2_row_t;

typedef struct {
    int valid;
    int stage, capacity, runnable, beff, contiguous;
} v2_batch_t;

typedef struct {
    int active;
    int stage, capacity, runnable, beff, contiguous;
    int weight, operation, path, leaf, reason;
    long long macs;
} v2_call_t;

static v2_row_t g_rows[V2_MAX_ROWS];
static atomic_int g_nrows;
static atomic_int g_enabled = -1;
static atomic_int g_overflow;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread v2_batch_t t_batch;
static __thread v2_call_t t_call;

static const char *const g_stage[] = { "other", "talker", "cp", "decoder", "prefill" };
static const char *const g_weight[] = { "other", "int8", "q4", "bf16", "fp32" };
static const char *const g_op[] = { "other", "gemv", "matmat", "qkv", "conv", "sgemm" };
static const char *const g_reason[] = {
    "none", "solo", "ragged", "noncontiguous", "max_b", "shape", "env_disabled",
    "isa_unavailable", "not_compiled", "region_unavailable", "decoder_policy", "generic_fallback"
};

const char *qwen_v2_stage_name(int v) {
    return (v >= 0 && v < QWEN_V2_STAGE_COUNT) ? g_stage[v] : "other";
}
const char *qwen_v2_weight_name(int v) {
    return (v >= 0 && v < QWEN_V2_WEIGHT_COUNT) ? g_weight[v] : "other";
}
const char *qwen_v2_op_name(int v) {
    return (v >= 0 && v < QWEN_V2_OP_COUNT) ? g_op[v] : "other";
}
const char *qwen_v2_reason_name(int v) {
    return (v >= 0 && v < QWEN_V2_REASON_COUNT) ? g_reason[v] : "generic_fallback";
}

static void v2_atexit(void) { qwen_v2_census_report(NULL); }

int qwen_v2_census_enabled(void) {
    int v = atomic_load_explicit(&g_enabled, memory_order_relaxed);
    if (v < 0) {
        const char *e = getenv("QWEN_V2_CENSUS");
        v = (e && e[0] && e[0] != '0');
        static atomic_int registered;
        if (v && !atomic_exchange_explicit(&registered, 1, memory_order_relaxed))
            atexit(v2_atexit);
        atomic_store_explicit(&g_enabled, v, memory_order_release);
    }
    return v;
}

void qwen_v2_census_batch_begin(int stage, int capacity, int runnable,
                                int B_eff, int contiguous) {
    if (!qwen_v2_census_enabled()) return;
    t_batch.valid = 1;
    t_batch.stage = stage;
    t_batch.capacity = capacity > 0 ? capacity : B_eff;
    t_batch.runnable = runnable > 0 ? runnable : B_eff;
    t_batch.beff = B_eff > 0 ? B_eff : 1;
    t_batch.contiguous = contiguous ? 1 : 0;
}

void qwen_v2_census_batch_end(void) {
    if (!qwen_v2_census_enabled()) return;
    t_batch.valid = 0;
}

int qwen_v2_census_batch_reason(int base_reason) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) <= 0 || !t_batch.valid)
        return base_reason;
    if (base_reason == QWEN_V2_REASON_NONE && t_batch.beff != t_batch.capacity)
        return QWEN_V2_REASON_RAGGED;
    return base_reason;
}

int qwen_v2_census_batch_width(void) {
    return atomic_load_explicit(&g_enabled, memory_order_acquire) > 0 && t_batch.valid
         ? t_batch.beff : 0;
}

void qwen_v2_census_call_begin(int stage, int weight, int operation, int reason,
                               int path_hint, int leaf_hint) {
    if (!qwen_v2_census_enabled()) return;
    memset(&t_call, 0, sizeof t_call);
    t_call.active = 1;
    t_call.stage = stage;
    t_call.capacity = t_batch.valid ? t_batch.capacity : 1;
    t_call.runnable = t_batch.valid ? t_batch.runnable : 1;
    t_call.beff = t_batch.valid ? t_batch.beff : 1;
    t_call.contiguous = t_batch.valid ? t_batch.contiguous : 1;
    t_call.weight = weight;
    t_call.operation = operation;
    t_call.reason = reason;
    t_call.path = path_hint;
    t_call.leaf = leaf_hint;
}

void qwen_v2_census_call_set_width(int B_eff) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) > 0 && t_call.active && B_eff > 0)
        t_call.beff = B_eff;
}

void qwen_v2_census_call_set_reason(int reason) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) > 0 && t_call.active)
        t_call.reason = reason;
}

void qwen_v2_census_note_path(int path, int rows, int cols, int B) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) <= 0 || !t_call.active) return;
    t_call.path = path;
    if (rows > 0 && cols > 0 && B > 0)
        t_call.macs += (long long)rows * (long long)cols * (long long)B;
}

void qwen_v2_census_note_leaf(int leaf) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) <= 0 || !t_call.active) return;
    t_call.leaf = leaf;
}

static int v2_same(const v2_row_t *r, const v2_call_t *c) {
    return r->stage == c->stage && r->capacity == c->capacity &&
           r->runnable == c->runnable && r->beff == c->beff &&
           r->contiguous == c->contiguous && r->weight == c->weight &&
           r->operation == c->operation && r->path == c->path &&
           r->leaf == c->leaf && r->reason == c->reason;
}

void qwen_v2_census_call_end(void) {
    if (atomic_load_explicit(&g_enabled, memory_order_acquire) <= 0 || !t_call.active) return;
    v2_call_t c = t_call;
    t_call.active = 0;
    int n = atomic_load_explicit(&g_nrows, memory_order_acquire);
    v2_row_t *hit = NULL;
    for (int i = 0; i < n; i++) if (v2_same(&g_rows[i], &c)) { hit = &g_rows[i]; break; }
    if (!hit) {
        pthread_mutex_lock(&g_mu);
        n = atomic_load_explicit(&g_nrows, memory_order_relaxed);
        for (int i = 0; i < n && !hit; i++) if (v2_same(&g_rows[i], &c)) hit = &g_rows[i];
        if (!hit && n < V2_MAX_ROWS) {
            hit = &g_rows[n];
            hit->stage = c.stage; hit->capacity = c.capacity; hit->runnable = c.runnable;
            hit->beff = c.beff; hit->contiguous = c.contiguous;
            hit->weight = c.weight; hit->operation = c.operation; hit->path = c.path;
            hit->leaf = c.leaf; hit->reason = c.reason;
            atomic_store_explicit(&g_nrows, n + 1, memory_order_release);
        } else if (!hit) {
            atomic_fetch_add_explicit(&g_overflow, 1, memory_order_relaxed);
        }
        pthread_mutex_unlock(&g_mu);
    }
    if (hit) {
        atomic_fetch_add_explicit(&hit->count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&hit->macs, c.macs, memory_order_relaxed);
    }
}

static int v2_cmp(const void *a, const void *b) {
    const v2_row_t *x = *(const v2_row_t *const *)a;
    const v2_row_t *y = *(const v2_row_t *const *)b;
#define CMP(f) do { if (x->f != y->f) return x->f < y->f ? -1 : 1; } while (0)
    CMP(stage); CMP(capacity); CMP(runnable); CMP(beff); CMP(contiguous);
    CMP(weight); CMP(operation); CMP(path); CMP(leaf); CMP(reason);
#undef CMP
    return 0;
}

void qwen_v2_census_report(void *out) {
    if (!qwen_v2_census_enabled()) return;
    FILE *f = out ? (FILE *)out : stderr;
    int n = atomic_load_explicit(&g_nrows, memory_order_acquire);
    v2_row_t *order[V2_MAX_ROWS];
    for (int i = 0; i < n; i++) order[i] = &g_rows[i];
    qsort(order, (size_t)n, sizeof order[0], v2_cmp);
    fprintf(f, "\n[v2-census] rows=%d overflow=%d\n", n,
            atomic_load_explicit(&g_overflow, memory_order_relaxed));
    fprintf(f, "# csv: stage,C,runnable,B_eff,contiguous,weight,operation,path,leaf,fallback_reason,count,macs\n");
    for (int i = 0; i < n; i++) {
        const v2_row_t *r = order[i];
        fprintf(f, "v2,%s,%d,%d,%d,%d,%s,%s,%s,%s,%s,%lld,%lld\n",
                qwen_v2_stage_name(r->stage), r->capacity, r->runnable, r->beff,
                r->contiguous, qwen_v2_weight_name(r->weight), qwen_v2_op_name(r->operation),
                qwen_path_name(r->path), qwen_leaf_name(r->leaf), qwen_v2_reason_name(r->reason),
                (long long)atomic_load_explicit(&r->count, memory_order_relaxed),
                (long long)atomic_load_explicit(&r->macs, memory_order_relaxed));
    }
    fflush(f);

    const char *jp = getenv("QWEN_V2_CENSUS_JSON");
    if (jp && jp[0]) {
        char path[1024];
        const char *pct = strstr(jp, "%d");
        if (pct) snprintf(path, sizeof path, "%.*s%d%s", (int)(pct - jp), jp,
                          (int)getpid(), pct + 2);
        else snprintf(path, sizeof path, "%s", jp);
        FILE *j = fopen(path, "w");
        if (j) {
            fprintf(j, "{\n  \"version\": 1,\n  \"rows\": [\n");
            for (int i = 0; i < n; i++) {
                const v2_row_t *r = order[i];
                fprintf(j, "    {\"stage\":\"%s\",\"capacity_C\":%d,\"runnable\":%d,"
                           "\"B_eff\":%d,\"contiguous\":%d,\"weight\":\"%s\","
                           "\"operation\":\"%s\",\"path\":\"%s\",\"leaf\":\"%s\","
                           "\"fallback_reason\":\"%s\",\"count\":%lld,\"macs\":%lld}%s\n",
                        qwen_v2_stage_name(r->stage), r->capacity, r->runnable, r->beff,
                        r->contiguous, qwen_v2_weight_name(r->weight), qwen_v2_op_name(r->operation),
                        qwen_path_name(r->path), qwen_leaf_name(r->leaf), qwen_v2_reason_name(r->reason),
                        (long long)atomic_load_explicit(&r->count, memory_order_relaxed),
                        (long long)atomic_load_explicit(&r->macs, memory_order_relaxed),
                        i + 1 < n ? "," : "");
            }
            fprintf(j, "  ]\n}\n");
            fclose(j);
        }
    }
}
