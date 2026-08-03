/* A/B gate: the qwen-tts safetensors reader against ingot, both linked in
 * one binary. Written BEFORE the migration touches a line; deleted when the
 * branch merges.
 *
 * This repo has the REAL checkpoints on disk, so the gate runs on them from
 * day one: QWEN_PARITY_DIR overrides the model directory (default
 * qwen3-tts-0.6b). Per tensor it compares name/dtype/ndim/shape/size, the
 * head AND tail 4 KiB of the raw payload (head catches a wrong offset, tail
 * a wrong length), and the f32 conversion path bit for bit on a sample.
 *
 * The count check matters most: the old reader stops at
 * SAFETENSORS_MAX_TENSORS (1024) and SAFETENSORS_MAX_SHARDS (8) WITHOUT an
 * error. If it reports exactly 1024 and ingot more, the model was silently
 * loaded in half. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "../qwen_tts_safetensors.h"
#include "ingot/safetensors.h"

static int failures, checks;
#define CHECK(cond, ...) do {                                        \
    checks++;                                                        \
    if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__);          \
                   printf("  (%s:%d)\n", __FILE__, __LINE__);        \
                   failures++; }                                     \
    else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); }  \
} while (0)

/* The two dtype vocabularies, aligned by meaning rather than by value. */
static ingot_dtype dtype_bridge(safetensor_dtype_t d) {
    switch (d) {
    case DTYPE_F32:  return INGOT_DT_F32;
    case DTYPE_F16:  return INGOT_DT_F16;
    case DTYPE_BF16: return INGOT_DT_BF16;
    case DTYPE_I32:  return INGOT_DT_I32;
    case DTYPE_I64:  return INGOT_DT_I64;
    case DTYPE_BOOL: return INGOT_DT_BOOL;
    default:         return INGOT_DT_UNKNOWN;
    }
}

int main(void) {
    const char *dir = getenv("QWEN_PARITY_DIR");
    if (dir == NULL || dir[0] == '\0') dir = "qwen3-tts-0.6b";

    printf("parity on %s\n", dir);
    multi_safetensors_t *ms = multi_safetensors_open(dir);
    ingot_st *st = NULL;
    char err[256] = "";
    CHECK(ms != NULL, "old reader opens the directory");
    CHECK(ingot_st_open_dir(&st, dir, err, sizeof err) == 0,
          "ingot opens it (%s)", err);
    if (ms == NULL || st == NULL) {
        printf("\nmodel directory missing? set QWEN_PARITY_DIR\n");
        return 1;
    }

    size_t old_count = 0;
    for (int s = 0; s < ms->num_shards; s++)
        old_count += (size_t)ms->shards[s]->num_tensors;
    CHECK(old_count == ingot_st_count(st),
          "same tensor count (%zu) — old reader caps at 1024/shard SILENTLY",
          old_count);
    if (old_count == 1024 * (size_t)ms->num_shards)
        printf("  ⚠ old count is exactly the cap: the model was truncated\n");

    size_t bad = 0, compared = 0;
    int f32_checked = 0;
    for (int s = 0; s < ms->num_shards; s++) {
        const safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            const safetensor_t *ot = &sf->tensors[i];
            const ingot_st_tensor *it = ingot_st_find(st, ot->name);
            if (it == NULL || dtype_bridge(ot->dtype) != it->dtype ||
                (uint32_t)ot->ndim != it->rank ||
                ot->data_size != it->nbytes) { bad++; continue; }
            int same = 1;
            for (int d = 0; d < ot->ndim; d++)
                if ((uint64_t)ot->shape[d] != it->shape[d]) same = 0;
            const unsigned char *od = safetensors_data(sf, ot);
            const unsigned char *id = ingot_st_data(st, it);
            const size_t head = ot->data_size < 4096 ? ot->data_size : 4096;
            same = same && od && id && memcmp(od, id, head) == 0 &&
                   memcmp(od + ot->data_size - head,
                          id + it->nbytes - head, head) == 0;
            if (!same) { bad++; continue; }
            compared++;
            /* f32 conversion parity, spot-checked on the first BF16 met:
             * safetensors_get_f32 (old, allocates) against ingot_st_to_f32
             * (buffer ours) — bf16 widening is exact, so bit-identical. */
            if (!f32_checked && ot->dtype == DTYPE_BF16 &&
                safetensor_numel(ot) <= (1 << 22)) {
                float *a = safetensors_get_f32(sf, ot);
                float *b = malloc((size_t)it->nelem * sizeof(float));
                CHECK(a != NULL && b != NULL &&
                          ingot_st_to_f32(st, it, b) == 0 &&
                          memcmp(a, b, (size_t)it->nelem * sizeof(float)) == 0,
                      "f32 conversion bit-identical on %s (%llu elems)",
                      ot->name, (unsigned long long)it->nelem);
                free(a); free(b);
                f32_checked = 1;
            }
        }
    }
    CHECK(bad == 0, "%zu tensors: dtype/shape/size + head&tail 4 KiB agree (%zu bad)",
          compared, bad);

    multi_safetensors_close(ms);
    ingot_st_close(st);
    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
