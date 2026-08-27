/* Type-generic entry points.
 *
 * The specialized kernels (ingot_q4_k_matvec and friends) require the caller
 * to know the format at the call site. That is right for an engine that has
 * decided its weights are Q4_K, and wrong for everything else: a loader that
 * opens whatever GGUF it is handed needs one call that works for any type it
 * can decode.
 *
 * So: dispatch to the hand-written kernel when there is one, and fall back to
 * decode-a-row-then-dot otherwise. The fallback is not slow in the way people
 * expect — it decodes each row ONCE per call, which is the same complexity as
 * the specialized matvec; what it gives up is the SIMD inner loop and, for
 * matmat, the int8 activation path.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/quant.h"
#include "internal.h"

void ingot_parallel_for(size_t count, ingot_range_fn fn, void *user);

typedef int (*matvec_fn)(const void *, size_t, size_t, const float *, float *);
typedef int (*matmat_fn)(const void *, size_t, size_t, const float *, float *, size_t);
typedef int (*deqmat_fn)(const void *, size_t, size_t, float *);

static void specialized(int type, matvec_fn *mv, matmat_fn *mm, deqmat_fn *dq) {
    *mv = NULL; *mm = NULL; *dq = NULL;
    switch (type) {
    case INGOT_TYPE_Q2_K: *mv = ingot_q2_k_matvec; *mm = ingot_q2_k_matmat; *dq = ingot_q2_k_dequant; break;
    case INGOT_TYPE_Q3_K: *mv = ingot_q3_k_matvec; *mm = ingot_q3_k_matmat; *dq = ingot_q3_k_dequant; break;
    case INGOT_TYPE_Q4_K: *mv = ingot_q4_k_matvec; *mm = ingot_q4_k_matmat; *dq = ingot_q4_k_dequant; break;
    case INGOT_TYPE_Q5_K: *mv = ingot_q5_k_matvec; *mm = ingot_q5_k_matmat; *dq = ingot_q5_k_dequant; break;
    case INGOT_TYPE_Q6_K: *mv = ingot_q6_k_matvec; *mm = ingot_q6_k_matmat; *dq = ingot_q6_k_dequant; break;
    case INGOT_TYPE_Q4_0: *mv = ingot_q4_0_matvec; *dq = ingot_q4_0_dequant; break;
    case INGOT_TYPE_Q8_0: *mv = ingot_q8_0_matvec; *mm = ingot_q8_0_matmat; *dq = ingot_q8_0_dequant; break;
    case INGOT_TYPE_BF16: *mv = ingot_bf16_matvec; *mm = ingot_bf16_matmat; *dq = ingot_bf16_dequant; break;
    case INGOT_TYPE_F16:  *mv = ingot_f16_matvec;  *mm = ingot_f16_matmat;  *dq = ingot_f16_dequant;  break;
    default: break;
    }
}

int ingot_has_kernel(int type) {
    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    return mv != NULL;
}

/* Row geometry shared by every generic path. Returns -1 on anything the type
 * cannot represent: unknown type, a row that is not a whole number of blocks,
 * or an overflow. */
static int row_geometry(int type, size_t rows, size_t cols, size_t *row_bytes) {
    uint64_t blk_elems, blk_bytes;
    if (rows == 0 || cols == 0 || ingot_type_geometry(type, &blk_elems, &blk_bytes) != 0)
        return -1;
    if (cols % blk_elems != 0) return -1;
    const uint64_t blocks = cols / blk_elems;
    if (blocks > UINT64_MAX / blk_bytes) return -1;
    const uint64_t bytes = blocks * blk_bytes;
    if (bytes > SIZE_MAX || rows > SIZE_MAX / (size_t)bytes) return -1;
    if (rows > SIZE_MAX / cols) return -1;
    *row_bytes = (size_t)bytes;
    return 0;
}

typedef struct {
    int type;
    const unsigned char *w;
    size_t rows, cols, row_bytes, tokens;
    const float *x;
    float *y;
    int failed;
} job;

/* One scratch row per worker range, not per row: the allocation is amortised
 * over the whole range, and a failure is recorded rather than thrown, because
 * there is nothing to throw to from inside a parallel_for. */
static void generic_range(size_t begin, size_t end, void *user) {
    job *j = (job *)user;
    float *row = (float *)malloc(j->cols * sizeof(float));
    if (row == NULL) { j->failed = 1; return; }
    for (size_t r = begin; r < end && !j->failed; r++) {
        if (ingot_dequant(j->type, j->w + r * j->row_bytes, j->cols, row) != 0) {
            j->failed = 1;
            break;
        }
        for (size_t t = 0; t < j->tokens; t++) {
            const float *x = j->x + t * j->cols;
            float sum = 0.0f;
            for (size_t i = 0; i < j->cols; i++) sum += row[i] * x[i];
            j->y[t * j->rows + r] = sum;
        }
    }
    free(row);
}

static void dequant_range(size_t begin, size_t end, void *user) {
    job *j = (job *)user;
    for (size_t r = begin; r < end && !j->failed; r++)
        if (ingot_dequant(j->type, j->w + r * j->row_bytes, j->cols,
                          j->y + r * j->cols) != 0) j->failed = 1;
}

int ingot_matvec(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output) {
    return ingot_matmat(type, weights, rows, cols, input, output, 1);
}

int ingot_matmat(int type, const void *weights, size_t rows, size_t cols,
                 const float *input, float *output, size_t tokens) {
    if (weights == NULL || input == NULL || output == NULL || tokens == 0) return -1;
    size_t row_bytes;
    if (row_geometry(type, rows, cols, &row_bytes) != 0) return -1;
    if (tokens > SIZE_MAX / rows) return -1;

    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    if (mm != NULL) return mm(weights, rows, cols, input, output, tokens);
    if (mv != NULL && tokens == 1) return mv(weights, rows, cols, input, output);
    if (!ingot_type_can_dequant(type)) return -1;

    job j = { type, (const unsigned char *)weights, rows, cols, row_bytes,
              tokens, input, output, 0 };
    ingot_parallel_for(rows, generic_range, &j);
    return j.failed ? -1 : 0;
}

int ingot_dequant_matrix(int type, const void *weights, size_t rows, size_t cols,
                         float *output) {
    if (weights == NULL || output == NULL) return -1;
    size_t row_bytes;
    if (row_geometry(type, rows, cols, &row_bytes) != 0) return -1;

    matvec_fn mv; matmat_fn mm; deqmat_fn dq;
    specialized(type, &mv, &mm, &dq);
    if (dq != NULL) return dq(weights, rows, cols, output);
    if (!ingot_type_can_dequant(type)) return -1;

    job j = { type, (const unsigned char *)weights, rows, cols, row_bytes,
              1, NULL, output, 0 };
    ingot_parallel_for(rows, dequant_range, &j);
    return j.failed ? -1 : 0;
}

/* ── straight off a GGUF tensor ─────────────────────────────────────────────
 * ggml stores ne[0] as the fastest dimension, which for a 2D weight matrix is
 * the input width. So rows = ne[1] and cols = ne[0] — the flip that every
 * consumer otherwise writes out by hand at each call site, and gets wrong
 * once. Rank 1 is treated as a single row. */
static int tensor_shape(const ingot_tensor *t, size_t *rows, size_t *cols) {
    if (t == NULL || t->rank == 0 || t->rank > 2) return -1;
    *cols = (size_t)t->ne[0];
    *rows = (t->rank == 2) ? (size_t)t->ne[1] : 1;
    return (*rows != 0 && *cols != 0) ? 0 : -1;
}

int ingot_gguf_matvec(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output) {
    return ingot_gguf_matmat(g, t, input, output, 1);
}

int ingot_gguf_matmat(const ingot_gguf *g, const ingot_tensor *t,
                      const float *input, float *output, size_t tokens) {
    size_t rows, cols;
    const void *w = ingot_gguf_data(g, t);
    if (w == NULL || tensor_shape(t, &rows, &cols) != 0) return -1;
    return ingot_matmat(t->type, w, rows, cols, input, output, tokens);
}

int ingot_gguf_dequant_matrix(const ingot_gguf *g, const ingot_tensor *t, float *output) {
    size_t rows, cols;
    const void *w = ingot_gguf_data(g, t);
    if (w == NULL || tensor_shape(t, &rows, &cols) != 0) return -1;
    return ingot_dequant_matrix(t->type, w, rows, cols, output);
}
