/* qwen_tts_costmap.c — see qwen_tts_costmap.h for the semantics.
 *
 * The whole point of this file is that it must not change what it measures.  Hence:
 * thread-local accumulation, no atomic read-modify-write on the hot path, one global
 * mutex touched once per thread (registration) and once per dump, and a gate that
 * costs a load and a branch when the map is off.
 */
#include "qwen_tts_costmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <stdint.h>

int qwen_costmap_level_v = 0;

typedef struct {
    int         id;
    const char *name;
    int         parent;
    int         level;
    const char *component;
    const char *mode;      /* "stack" = begin/end pair; "derived" = added duration */
} rgn_info_t;

/* Append-only.  Anything not listed is reported as "region/<id>" with no parent. */
static const rgn_info_t g_rgn[] = {
    { QWEN_RGN_TK_PREFILL,        "talker.prefill.total",             QWEN_RGN_NONE,        1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_NORM,        "talker.prefill.norm",              QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_QKV,         "talker.prefill.qkv",               QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_QKROPEKV,    "talker.prefill.qknorm_rope_kv",    QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_ATTN,        "talker.prefill.attention",         QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_OPROJ,       "talker.prefill.out_proj",          QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_GATEUP,      "talker.prefill.gate_up",           QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_FFN_ACT,     "talker.prefill.ffn_act",           QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_DOWN,        "talker.prefill.down",              QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_WEIGHT_PREP, "talker.prefill.weight_prep",       QWEN_RGN_TK_PREFILL,  1, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_LAYOUT_IN,   "talker.prefill.layout_in",         QWEN_RGN_MULTI,       2, "talker"  , "stack" },
    { QWEN_RGN_TK_PF_LAYOUT_OUT,  "talker.prefill.layout_out",        QWEN_RGN_MULTI,       2, "talker"  , "stack" },
    { QWEN_RGN_TK_DECODE,         "talker.decode.total",              QWEN_RGN_NONE,        1, "talker"  , "stack" },

    { QWEN_RGN_CP_PREFILL,        "cp.prefill.total",                 QWEN_RGN_CP_DECODE,   1, "cp"      , "stack" },
    { QWEN_RGN_CP_DECODE,         "cp.decode.total",                  QWEN_RGN_NONE,        1, "cp"      , "stack" },
    { QWEN_RGN_CP_D_QKV,          "cp.decode.qkv",                    QWEN_RGN_CP_DECODE,   2, "cp"      , "stack" },
    { QWEN_RGN_CP_D_ATTN,         "cp.decode.attention",              QWEN_RGN_CP_DECODE,   2, "cp"      , "stack" },
    { QWEN_RGN_CP_D_OPROJ,        "cp.decode.out_proj",               QWEN_RGN_CP_DECODE,   2, "cp"      , "stack" },
    { QWEN_RGN_CP_D_GATEUP,       "cp.decode.gate_up",                QWEN_RGN_CP_DECODE,   2, "cp"      , "stack" },
    { QWEN_RGN_CP_D_DOWN,         "cp.decode.down",                   QWEN_RGN_CP_DECODE,   2, "cp"      , "stack" },
    { QWEN_RGN_CP_D_LMHEAD,       "cp.decode.lm_head",                QWEN_RGN_CP_DECODE,   1, "cp"      , "stack" },

    { QWEN_RGN_SD_TOTAL,          "decoder.total",                    QWEN_RGN_NONE,        1, "decoder" , "stack" },
    { QWEN_RGN_SD_VQ,             "decoder.vq",                       QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_PRECONV,        "decoder.pre_conv",                 QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_INPROJ,         "decoder.in_proj",                  QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_TRANSFORMER,    "decoder.transformer",              QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_OUTPROJ,        "decoder.out_proj",                 QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_CONVSTACK,      "decoder.conv_stack",               QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },
    { QWEN_RGN_SD_POST,           "decoder.post",                     QWEN_RGN_SD_TOTAL,    1, "decoder" , "stack" },

    { QWEN_RGN_RT_REQUEST,        "runtime.request.total",            QWEN_RGN_NONE,        1, "runtime" , "derived" },
    { QWEN_RGN_RT_ADMISSION,      "runtime.admission",                QWEN_RGN_RT_REQUEST,  1, "runtime" , "derived" },
    { QWEN_RGN_RT_POOL_DISPATCH,  "runtime.pool_dispatch",            QWEN_RGN_MULTI,       1, "runtime" , "stack" },
    { QWEN_RGN_RT_POOL_WAIT,      "runtime.pool_wait_completion",     QWEN_RGN_RT_POOL_DISPATCH, 1, "runtime" , "stack" },
    { QWEN_RGN_RT_POOL_SUBMIT,    "runtime.pool_submit_wait",         QWEN_RGN_RT_POOL_DISPATCH, 1, "runtime" , "stack" },
    { QWEN_RGN_SD_CONV_INT8,      "decoder.conv_int8.panels",         QWEN_RGN_MULTI,       1, "decoder" , "stack" },
    { QWEN_RGN_MM_REGION_I8,      "region.int8_runner.rows",          QWEN_RGN_MULTI,       1, "runtime" , "stack" },
};
static const int g_rgn_n = (int)(sizeof g_rgn / sizeof g_rgn[0]);

static const rgn_info_t *rgn_find(int id) {
    for (int i = 0; i < g_rgn_n; i++) if (g_rgn[i].id == id) return &g_rgn[i];
    return NULL;
}
const char *qwen_region_name(int id)      { const rgn_info_t *r = rgn_find(id); return r ? r->name : "?"; }
int         qwen_region_parent(int id)    { const rgn_info_t *r = rgn_find(id); return r ? r->parent : QWEN_RGN_NONE; }
int         qwen_region_level(int id)     { const rgn_info_t *r = rgn_find(id); return r ? r->level : 1; }
const char *qwen_region_component(int id) { const rgn_info_t *r = rgn_find(id); return r ? r->component : "other"; }
static const char *rgn_mode(int id) { const rgn_info_t *r = rgn_find(id); return r ? r->mode : "stack"; }

/* ---------------------------------------------------------------------------- */

#define RGN_STACK_MAX 24

typedef struct rgn_tls_s {
    uint64_t ns[QWEN_RGN_MAX];
    uint64_t child_ns[QWEN_RGN_MAX];
    uint64_t calls[QWEN_RGN_MAX];
    uint32_t nest_mismatch[QWEN_RGN_MAX];
    uint64_t units[QWEN_RGN_MAX];        /* work units this thread claimed        */
    uint64_t tasks[QWEN_RGN_MAX];        /* units offered by dispatches it opened */
    uint32_t pool_nt[QWEN_RGN_MAX];      /* widest dispatch seen for this region  */
    uint64_t dispatches[QWEN_RGN_MAX];
    uint64_t entered[QWEN_RGN_MAX];      /* workers that entered the job body   */
    int      stack_id[RGN_STACK_MAX];
    uint64_t stack_t0[RGN_STACK_MAX];
    int      depth;
    uint64_t overflow;     /* begins dropped because the stack was full  */
    uint64_t unbalanced;   /* ends that did not match the top of stack   */
    uint64_t leaked;       /* regions closed by an unwind, not by an end */
    char     role[24];
    long     tid;
    struct rgn_tls_s *next;
} rgn_tls_t;

static __thread rgn_tls_t *t_rgn;
static rgn_tls_t         *g_rgn_list;
static pthread_mutex_t    g_rgn_mtx = PTHREAD_MUTEX_INITIALIZER;
static uint64_t           g_requests;          /* only touched at request end */
static pthread_mutex_t    g_req_mtx = PTHREAD_MUTEX_INITIALIZER;

static inline uint64_t rgn_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static rgn_tls_t *rgn_self(void) {
    rgn_tls_t *t = t_rgn;
    if (t) return t;
    t = (rgn_tls_t *)calloc(1, sizeof *t);
    if (!t) return NULL;
    snprintf(t->role, sizeof t->role, "%s", "unlabelled");
    t->tid = (long)(uintptr_t)pthread_self();
    pthread_mutex_lock(&g_rgn_mtx);
    t->next = g_rgn_list;
    g_rgn_list = t;
    pthread_mutex_unlock(&g_rgn_mtx);
    t_rgn = t;
    return t;
}

void qwen_costmap_init(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char *e = getenv("QWEN_COST_MAP");
    int lv = 0;
    if (e && e[0] && e[0] != '0') lv = (e[0] == '2') ? 2 : 1;
    qwen_costmap_level_v = lv;
}
int qwen_costmap_level(void) { return qwen_costmap_level_v; }

void qwen_region_pool_at_(int id, int threads, long long tasks) {
    if (id <= 0 || id >= QWEN_RGN_MAX) return;
    rgn_tls_t *t = rgn_self();
    if (!t) return;
    t->dispatches[id]++;
    t->tasks[id] += (uint64_t)(tasks > 0 ? tasks : 0);
    if (threads > 0 && (uint32_t)threads > t->pool_nt[id]) t->pool_nt[id] = (uint32_t)threads;
}

void qwen_region_workers_at_(int id, int entered) {
    if (id <= 0 || id >= QWEN_RGN_MAX || entered <= 0) return;
    rgn_tls_t *t = rgn_self();
    if (!t) return;
    t->entered[id] += (uint64_t)entered;
}

void qwen_region_units_at_(int id, long long n) {
    if (id <= 0 || id >= QWEN_RGN_MAX || n <= 0) return;
    rgn_tls_t *t = rgn_self();
    if (!t) return;
    t->units[id] += (uint64_t)n;
}

void qwen_region_thread_role(const char *role) {
    if (!qwen_costmap_level_v || !role) return;
    rgn_tls_t *t = rgn_self();
    if (t) snprintf(t->role, sizeof t->role, "%s", role);
}

void qwen_region_begin_(int id) {
    rgn_tls_t *t = rgn_self();
    if (!t || id <= 0 || id >= QWEN_RGN_MAX) return;
    if (t->depth >= RGN_STACK_MAX) { t->overflow++; return; }
    /* Verify the declared nesting instead of trusting it. */
    const int decl = qwen_region_parent(id);
    if (decl != QWEN_RGN_MULTI) {
        const int dyn = t->depth > 0 ? t->stack_id[t->depth - 1] : QWEN_RGN_NONE;
        if (dyn != decl) t->nest_mismatch[id]++;
    }
    t->stack_id[t->depth] = id;
    t->stack_t0[t->depth] = rgn_now_ns();
    t->depth++;
}

void qwen_region_end_(int id) {
    rgn_tls_t *t = t_rgn;
    if (!t || id <= 0 || id >= QWEN_RGN_MAX) return;
    if (t->depth <= 0 || t->stack_id[t->depth - 1] != id) { t->unbalanced++; return; }
    const uint64_t dt = rgn_now_ns() - t->stack_t0[t->depth - 1];
    t->depth--;
    t->ns[id] += dt;
    t->calls[id]++;
    if (t->depth > 0) t->child_ns[t->stack_id[t->depth - 1]] += dt;
}

int qwen_region_begin_unique_(int id) {
    rgn_tls_t *t = rgn_self();
    if (!t) return 0;
    for (int i = 0; i < t->depth; i++) if (t->stack_id[i] == id) return 0;
    qwen_region_begin_(id);
    return 1;
}

int qwen_region_depth(void) {
    rgn_tls_t *t = t_rgn;
    return t ? t->depth : 0;
}

void qwen_region_unwind(int depth) {
    rgn_tls_t *t = t_rgn;
    if (!t) return;
    while (t->depth > depth) {
        const int id = t->stack_id[t->depth - 1];
        const uint64_t dt = rgn_now_ns() - t->stack_t0[t->depth - 1];
        t->depth--;
        t->ns[id] += dt;
        t->calls[id]++;
        t->leaked++;
        if (t->depth > 0) t->child_ns[t->stack_id[t->depth - 1]] += dt;
    }
}

void qwen_costmap_after_fork(void) {
    if (!qwen_costmap_level_v) return;
    /* Single-threaded here by definition: fork() left one thread in the child. */
    for (rgn_tls_t *t = g_rgn_list; t; t = t->next) {
        memset(t->ns, 0, sizeof t->ns);
        memset(t->child_ns, 0, sizeof t->child_ns);
        memset(t->calls, 0, sizeof t->calls);
        memset(t->nest_mismatch, 0, sizeof t->nest_mismatch);
        t->depth = 0; t->overflow = t->unbalanced = t->leaked = 0;
    }
    g_requests = 0;
    pthread_mutex_init(&g_rgn_mtx, NULL);
    pthread_mutex_init(&g_req_mtx, NULL);
}

void qwen_region_add_ns(int id, unsigned long long ns) {
    if (!qwen_costmap_level_v) return;
    rgn_tls_t *t = rgn_self();
    if (!t || id <= 0 || id >= QWEN_RGN_MAX) return;
    t->ns[id] += ns;
    t->calls[id]++;
}

static int g_count_requests = 1;
void qwen_costmap_count_requests(int on) { g_count_requests = on; }

void qwen_costmap_request_done(void) {
    if (!qwen_costmap_level_v || !g_count_requests) return;
    pthread_mutex_lock(&g_req_mtx);
    g_requests++;
    pthread_mutex_unlock(&g_req_mtx);
}

/* ---------------------------------------------------------------------------- */

static void rgn_expand_pid(char *dst, size_t cap, const char *src) {
    size_t o = 0;
    for (size_t i = 0; src[i] && o + 1 < cap; i++) {
        if (src[i] == '%' && src[i + 1] == 'd') {
            o += (size_t)snprintf(dst + o, cap - o, "%d", (int)getpid());
            i++;
        } else dst[o++] = src[i];
    }
    dst[o < cap ? o : cap - 1] = '\0';
}

int qwen_costmap_dump(const char *path) {
    if (!qwen_costmap_level_v || !path || !*path) return 0;
    char real[1024];
    rgn_expand_pid(real, sizeof real, path);
    FILE *f = fopen(real, "w");
    if (!f) return -1;

    pthread_mutex_lock(&g_rgn_mtx);
    fprintf(f, "{\n \"v\": 1,\n \"pid\": %d,\n \"level\": %d,\n"
               " \"clock\": \"CLOCK_MONOTONIC\",\n \"semantics\": \"inclusive\",\n"
               " \"exclusive_rule\": \"self_ns = ns - child_ns\",\n"
               " \"requests\": %llu,\n \"threads\": [\n",
            (int)getpid(), qwen_costmap_level_v, (unsigned long long)g_requests);
    int first_t = 1;
    for (rgn_tls_t *t = g_rgn_list; t; t = t->next) {
        int any = 0;
        for (int i = 0; i < QWEN_RGN_MAX; i++) if (t->calls[i]) { any = 1; break; }
        if (!any) continue;
        if (!first_t) fprintf(f, ",\n");
        first_t = 0;
        fprintf(f, "  { \"role\": \"%s\", \"tid\": %ld, \"stack_overflow\": %llu, "
                   "\"unbalanced\": %llu, \"leaked\": %llu, \"regions\": [\n",
                t->role, t->tid, (unsigned long long)t->overflow,
                (unsigned long long)t->unbalanced, (unsigned long long)t->leaked);
        int first_r = 1;
        for (int i = 0; i < QWEN_RGN_MAX; i++) {
            if (!t->calls[i] && !t->units[i] && !t->dispatches[i] && !t->entered[i]) continue;
            if (!first_r) fprintf(f, ",\n");
            first_r = 0;
            fprintf(f, "   { \"id\": %d, \"name\": \"%s\", \"component\": \"%s\", "
                       "\"parent\": %d, \"level\": %d, \"mode\": \"%s\", \"calls\": %llu, "
                       "\"ns\": %llu, \"child_ns\": %llu, \"nest_mismatch\": %u, "
                       "\"units\": %llu, \"tasks\": %llu, \"dispatches\": %llu, "
                       "\"pool_threads\": %u, \"entered\": %llu }",
                    i, qwen_region_name(i), qwen_region_component(i),
                    qwen_region_parent(i), qwen_region_level(i), rgn_mode(i),
                    (unsigned long long)t->calls[i], (unsigned long long)t->ns[i],
                    (unsigned long long)t->child_ns[i], t->nest_mismatch[i],
                    (unsigned long long)t->units[i], (unsigned long long)t->tasks[i],
                    (unsigned long long)t->dispatches[i], t->pool_nt[i],
                    (unsigned long long)t->entered[i]);
        }
        fprintf(f, "\n  ] }");
    }
    fprintf(f, "\n ]\n}\n");
    pthread_mutex_unlock(&g_rgn_mtx);
    fclose(f);
    return 0;
}

void qwen_costmap_dump_env(void) {
    const char *p = getenv("QWEN_COSTMAP_JSON");
    if (p && *p) qwen_costmap_dump(p);
}

/* Resolve the level before main() so no thread can race the first marker, and make
 * a CLI run dump on the way out without every entry point having to remember to. */
__attribute__((constructor)) static void qwen_costmap_ctor(void) {
    qwen_costmap_init();
    if (qwen_costmap_level_v) {
        qwen_region_thread_role("main");
        atexit(qwen_costmap_dump_env);
    }
}
