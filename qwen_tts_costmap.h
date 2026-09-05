/* qwen_tts_costmap.h — profiler V1 step 2: coarse semantic cost map.
 *
 * ONE region layer with ONE timing semantics, so the numbers from the Talker, the
 * code predictor, the speech decoder and the server can finally be put in the same
 * table.  It answers: "outside and around the kernels the census already attributes,
 * where does the wall time of a component go?"
 *
 * SEMANTICS (fixed before any number was collected):
 *   - INCLUSIVE.  A region's time contains the time of every region entered inside it.
 *     Exclusive ("self") time is derived at report time as ns - child_ns, never measured
 *     separately, so inclusive and exclusive can never be silently mixed.
 *   - Nesting is declared statically (parent in the table) and VERIFIED at runtime: a
 *     begin whose dynamic parent is not the declared one bumps nest_mismatch for that
 *     region instead of being silently accepted.  QWEN_RGN_MULTI marks the regions that
 *     legitimately have several parents (the pool dispatch runs under every component).
 *   - Accumulation is THREAD-LOCAL.  No atomics on the hot path; the per-thread blocks
 *     are linked into a global list once per thread, under a mutex, at first use.
 *     Merging happens only at dump time.
 *   - CLOCK_MONOTONIC, nanoseconds, one clock read per begin and one per end.
 *
 * LEVELS (QWEN_COST_MAP):
 *   0 / unset  off.  One relaxed load of an int and a predictable branch per marker.
 *   1          macro regions only: component totals, Talker prefill stages, decoder
 *              stages, pool and server.  Nothing inside a per-frame or per-layer loop.
 *   2          adds the fine regions: per-layer CP decode stages and the Talker prefill
 *              layout transposes.  These sit in a hot loop (not in an inner kernel) and
 *              are opt-in precisely so level 1 can stay cheap.
 *
 * Ids are APPEND-ONLY: a report from an old binary must keep meaning.
 */
#ifndef QWEN_TTS_COSTMAP_H
#define QWEN_TTS_COSTMAP_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWEN_RGN_NONE = 0,

    /* ---- Talker ------------------------------------------------------------ */
    QWEN_RGN_TK_PREFILL = 1,      /* qwen_talker_prefill(), whole call            */
    QWEN_RGN_TK_PF_NORM,          /* input_norm + post_attn_norm (both sites)     */
    QWEN_RGN_TK_PF_QKV,           /* Q/K/V projection                             */
    QWEN_RGN_TK_PF_QKROPEKV,      /* q/k per-head norm + RoPE + KV cache store    */
    QWEN_RGN_TK_PF_ATTN,          /* causal attention                             */
    QWEN_RGN_TK_PF_OPROJ,         /* output projection + residual                 */
    QWEN_RGN_TK_PF_GATEUP,        /* fused gate/up projection                     */
    QWEN_RGN_TK_PF_FFN_ACT,       /* SwiGLU + compaction copy                     */
    QWEN_RGN_TK_PF_DOWN,          /* down projection + residual                   */
    QWEN_RGN_TK_PF_WEIGHT_PREP,   /* dequant to f32 on the non-matmat fallback    */
    QWEN_RGN_TK_PF_LAYOUT_IN,     /* [B][K] -> [K][B] activation transpose  (L2)  */
    QWEN_RGN_TK_PF_LAYOUT_OUT,    /* [N][B] -> [B][N] output transpose      (L2)  */
    QWEN_RGN_TK_DECODE,           /* one decode step, single or batched-ragged    */

    /* ---- Code predictor ---------------------------------------------------- */
    QWEN_RGN_CP_PREFILL = 20,     /* cp_prefill2(): a phase INSIDE the decode call */
    QWEN_RGN_CP_DECODE,           /* qwen_cp_predict / qwen_batch_cp_predict      */
    QWEN_RGN_CP_D_QKV,            /* per layer, per group                   (L2)  */
    QWEN_RGN_CP_D_ATTN,           /* qk-norm + RoPE + KV + attention        (L2)  */
    QWEN_RGN_CP_D_OPROJ,          /* output projection + residual norm      (L2)  */
    QWEN_RGN_CP_D_GATEUP,         /* gate/up projection + SwiGLU            (L2)  */
    QWEN_RGN_CP_D_DOWN,           /* down projection + residual norm        (L2)  */
    QWEN_RGN_CP_D_LMHEAD,         /* the 15 argmax matvecs                        */

    /* ---- Speech decoder ---------------------------------------------------- */
    QWEN_RGN_SD_TOTAL = 40,       /* any of the four decode entry points          */
    QWEN_RGN_SD_VQ,               /* codebook lookup + RVQ output projection      */
    QWEN_RGN_SD_PRECONV,          /* pre-conv                                     */
    QWEN_RGN_SD_INPROJ,           /* input projection                             */
    QWEN_RGN_SD_TRANSFORMER,      /* the decoder transformer stack                */
    QWEN_RGN_SD_OUTPROJ,          /* output projection                            */
    QWEN_RGN_SD_CONVSTACK,        /* ConvNeXt + upsampling conv stack             */
    QWEN_RGN_SD_POST,             /* final conv, windowing, audio emit            */

    /* ---- Runtime / server -------------------------------------------------- */
    QWEN_RGN_RT_REQUEST = 60,     /* recv -> completion, one per served request   */
    QWEN_RGN_RT_ADMISSION,        /* enqueue -> admit: time queued, not serving   */
    QWEN_RGN_RT_POOL_DISPATCH,    /* qwen_parallel(), inclusive (MULTI parent)    */
    QWEN_RGN_RT_POOL_WAIT,        /* caller done, waiting for workers to finish   */
    QWEN_RGN_RT_POOL_SUBMIT,      /* waiting to acquire the submit lock           */

    /* ---- parallel work decomposition (appended; ids are append-only) ------- */
    QWEN_RGN_SD_CONV_INT8 = 66,   /* decoder INT8 conv: panels claimed per worker */
    QWEN_RGN_MM_REGION_I8,        /* in-region INT8 runner: row blocks per worker */

    QWEN_RGN_MAX = 72
};

#define QWEN_RGN_MULTI (-1)       /* declared parent for legitimately multi-parent regions */

/* 0 = off, 1 = macro, 2 = macro + fine.  Read directly by the inline markers. */
extern int qwen_costmap_level_v;

void qwen_costmap_init(void);                 /* reads QWEN_COST_MAP once */
int  qwen_costmap_level(void);
void qwen_region_begin_(int id);
void qwen_region_end_(int id);

/* Depth of the calling thread's region stack, and "close everything opened above
 * this depth".  A component entry point records the depth right after opening its
 * own region and unwinds to it on the way out, so the dozen early `return`s inside
 * a decoder body cannot leave a region open forever.  Whatever gets unwound is
 * counted in `leaked` and shows up in the report instead of silently skewing it. */
int  qwen_region_depth(void);
void qwen_region_unwind(int depth);

/* Open `id` only if it is not already open on this thread; returns 1 when it opened.
 * The speech decoder has four public entry points and some of them delegate to each
 * other, so "decoder.total" must be owned by the outermost call or the total would
 * be counted twice.  Callers end the region only when this returned 1. */
int  qwen_region_begin_unique_(int id);
static inline int qwen_region_begin_unique(int id) {
    return qwen_costmap_level_v ? qwen_region_begin_unique_(id) : 0;
}

/* ---- pool occupancy: who actually did the work ---------------------------------
 *
 * Wall time alone cannot say that a region ran on two of six workers.  The decoder INT8
 * conv did exactly that for months -- its parallel unit was a fixed 128-column panel, so
 * the most expensive layer had two panels and four workers idled -- and finding it took a
 * manual dissection.  These record the decomposition itself:
 *
 *   qwen_region_pool_at(id, threads, tasks)   the dispatch: workers asked for, units offered
 *   qwen_region_units_at(id, n)               a worker claiming n units of that region
 *
 * The id is explicit because a pool worker runs on its own thread with its own region
 * stack: it is not "inside" the caller's region and cannot infer the attribution.
 * Accumulation stays thread-local; occupancy is derived at dump time as
 * (threads that claimed at least one unit) / (threads the dispatch asked for). */
void qwen_region_pool_at_(int id, int threads, long long tasks);
void qwen_region_units_at_(int id, long long n);
/* Workers that actually ENTERED the job body, counted by the job itself rather than inferred
 * from which threads happened to touch a marker: a thread that never reaches a marker leaves
 * no record, and turning that silence into an underfill claim would be a lie. */
void qwen_region_workers_at_(int id, int entered);
/* Count an event WITHOUT reading the clock.  FAST uses this where the event is frequent but
 * its duration is already inside a coarse region: the pool dispatch happens tens of thousands
 * of times in one request, and timestamping all three of its regions was 88% of all profiler
 * events and doubled TTFA.  DEEP still times them. */
void qwen_region_tick_at_(int id, long long n);
static inline void qwen_region_tick_at(int id, long long n) {
    if (qwen_costmap_level_v) qwen_region_tick_at_(id, n);
}
static inline void qwen_region_pool_at(int id, int threads, long long tasks) {
    if (qwen_costmap_level_v) qwen_region_pool_at_(id, threads, tasks);
}
static inline void qwen_region_units_at(int id, long long n) {
    if (qwen_costmap_level_v) qwen_region_units_at_(id, n);
}
/* DEEP-only variants.  Row-block and per-projection accounting is a micro-event: on x86 the
 * in-region runner alone produced 813k of these in one request, dwarfing everything else.
 * FAST keeps the coarse dispatch summary; DEEP gets the per-worker detail. */
static inline void qwen_region_units_at2(int id, long long n) {
    if (qwen_costmap_level_v > 1) qwen_region_units_at_(id, n);
}
static inline void qwen_region_workers_at2(int id, int entered) {
    if (qwen_costmap_level_v > 1) qwen_region_workers_at_(id, entered);
}
static inline void qwen_region_pool_at2(int id, int threads, long long tasks) {
    if (qwen_costmap_level_v > 1) qwen_region_pool_at_(id, threads, tasks);
}
static inline void qwen_region_workers_at(int id, int entered) {
    if (qwen_costmap_level_v) qwen_region_workers_at_(id, entered);
}

/* Label the calling thread for attribution ("main", "decoder", "prefill_helper", ...).
 * Purely descriptive; regions are accumulated per OS thread regardless. */
void qwen_region_thread_role(const char *role);

/* Add a duration that was NOT measured by a begin/end pair on one thread.  The
 * server request lifecycle is recorded as timestamps by threads that hand the job
 * over to each other, so a thread-local stack cannot bracket it; those regions are
 * marked mode="derived" in the JSON so nobody reads them as if they were measured
 * the same way as the rest. */
void qwen_region_add_ns(int id, unsigned long long ns);

/* One completed request, so the report can print ms/request. */
void qwen_costmap_request_done(void);

/* Stop counting completed requests for a while.  The server pre-warm runs a full
 * synthesis through qwen_tts_generate(), which is not a request anyone made: counting
 * it would dilute every ms/request in the report by one synthetic unit per worker. */
void qwen_costmap_count_requests(int on);

/* Drop everything accumulated before a fork.  A prefork server does its model load and
 * its pre-warm in the parent, so every worker would otherwise inherit that work and
 * report it again: merging N workers would count the same pre-warm N times.  After
 * this call a worker's cost map describes only what that worker actually served. */
void qwen_costmap_after_fork(void);

/* Write the merged cost map as JSON.  "%d" in the path is replaced by the pid, so a
 * prefork server can dump one file per worker exactly like the shape census. */
int  qwen_costmap_dump(const char *path);
/* Dump to QWEN_COSTMAP_JSON if it is set; called from the census dump path. */
void qwen_costmap_dump_env(void);

/* Static taxonomy, for the report tool and for --cost-map. */
const char *qwen_region_name(int id);
int         qwen_region_parent(int id);
int         qwen_region_level(int id);
const char *qwen_region_component(int id);

static inline void qwen_region_begin(int id) {
    if (qwen_costmap_level_v) qwen_region_begin_(id);
}
static inline void qwen_region_end(int id) {
    if (qwen_costmap_level_v) qwen_region_end_(id);
}
/* Fine (level 2) markers: separate entry points so a level-1 run never even reaches
 * the call, which is what keeps the per-layer sites off the level-1 cost. */
static inline void qwen_region_begin2(int id) {
    if (qwen_costmap_level_v > 1) qwen_region_begin_(id);
}
static inline void qwen_region_end2(int id) {
    if (qwen_costmap_level_v > 1) qwen_region_end_(id);
}

#ifdef __cplusplus
}
#endif
#endif /* QWEN_TTS_COSTMAP_H */
