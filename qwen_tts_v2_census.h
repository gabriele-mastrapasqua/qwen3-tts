/* qwen_tts_v2_census.h - default-off execution census for the v2 ragged server. */
#ifndef QWEN_TTS_V2_CENSUS_H
#define QWEN_TTS_V2_CENSUS_H

/* Keep these ids stable: qualification tooling stores them in machine-readable output. */
enum {
    QWEN_V2_STAGE_OTHER = 0,
    QWEN_V2_STAGE_TALKER,
    QWEN_V2_STAGE_CP,
    QWEN_V2_STAGE_DECODER,
    QWEN_V2_STAGE_PREFILL,
    QWEN_V2_STAGE_COUNT
};
enum {
    QWEN_V2_WEIGHT_OTHER = 0,
    QWEN_V2_WEIGHT_INT8,
    QWEN_V2_WEIGHT_Q4,
    QWEN_V2_WEIGHT_BF16,
    QWEN_V2_WEIGHT_FP32,
    QWEN_V2_WEIGHT_COUNT
};
enum {
    QWEN_V2_OP_OTHER = 0,
    QWEN_V2_OP_GEMV,
    QWEN_V2_OP_MATMAT,
    QWEN_V2_OP_QKV,
    QWEN_V2_OP_CONV,
    QWEN_V2_OP_SGEMM,
    QWEN_V2_OP_COUNT
};
enum {
    QWEN_V2_REASON_NONE = 0,
    QWEN_V2_REASON_SOLO,
    QWEN_V2_REASON_RAGGED,
    QWEN_V2_REASON_NONCONTIGUOUS,
    QWEN_V2_REASON_MAX_B,
    QWEN_V2_REASON_SHAPE,
    QWEN_V2_REASON_ENV_DISABLED,
    QWEN_V2_REASON_ISA_UNAVAILABLE,
    QWEN_V2_REASON_NOT_COMPILED,
    QWEN_V2_REASON_REGION_UNAVAILABLE,
    QWEN_V2_REASON_DECODER_POLICY,
    QWEN_V2_REASON_GENERIC_FALLBACK,
    QWEN_V2_REASON_COUNT
};

int  qwen_v2_census_enabled(void);

/* Set the request-level context.  C is the configured server capacity; B_eff is the
 * live runnable cohort that the next projection/decoder operation sees. */
void qwen_v2_census_batch_begin(int stage, int capacity, int runnable,
                                int B_eff, int contiguous);
void qwen_v2_census_batch_end(void);
int  qwen_v2_census_batch_reason(int base_reason);
int  qwen_v2_census_batch_width(void);

/* Attribute one dispatcher call to the current batch context.  path/leaf are hints;
 * qwen_census_op/leaf update them with the branch that actually ran. */
void qwen_v2_census_call_begin(int stage, int weight, int operation, int reason,
                               int path_hint, int leaf_hint);
void qwen_v2_census_call_set_width(int B_eff);
void qwen_v2_census_call_set_reason(int reason);
void qwen_v2_census_call_end(void);

/* Hooks used by the existing shape census.  They are no-ops when QWEN_V2_CENSUS is off. */
void qwen_v2_census_note_path(int path, int rows, int cols, int B);
void qwen_v2_census_note_leaf(int leaf);

void qwen_v2_census_report(void *out);

const char *qwen_v2_stage_name(int stage);
const char *qwen_v2_weight_name(int weight);
const char *qwen_v2_op_name(int operation);
const char *qwen_v2_reason_name(int reason);

#endif
