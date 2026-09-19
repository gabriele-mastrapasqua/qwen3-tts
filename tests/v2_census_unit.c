#include "qwen_tts_v2_census.h"
#include "qwen_tts_batch.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Production names come from qwen_tts_kernels.c.  Keep this test independent
 * of the full model binary so it runs on every build host. */
const char *qwen_path_name(int path) { return path == 20 ? "matmat_bf16" : "test_path"; }
const char *qwen_leaf_name(int leaf) { return leaf == 1 ? "vnni" : "none"; }

int main(void) {
    for (int b = 1; b <= 32; b++) {
        int p[8] = {0};
        int n = qwen_batch_chunk_plan(b, 16, p, 8);
        assert(n == (b + 15) / 16);
        int sum = 0;
        for (int i = 0; i < n; i++) { assert(p[i] >= 1 && p[i] <= 16); sum += p[i]; }
        assert(sum == b);
    }
    setenv("QWEN_V2_CENSUS", "1", 1);
    qwen_v2_census_batch_begin(QWEN_V2_STAGE_TALKER, 8, 3, 3, 0);
    qwen_v2_census_call_begin(QWEN_V2_STAGE_TALKER, QWEN_V2_WEIGHT_INT8,
                              QWEN_V2_OP_MATMAT, QWEN_V2_REASON_RAGGED, 20, 0);
    qwen_v2_census_note_path(20, 32, 64, 3);
    qwen_v2_census_note_leaf(1);
    qwen_v2_census_call_end();

    FILE *f = tmpfile();
    assert(f);
    qwen_v2_census_report(f);
    fflush(f);
    rewind(f);
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = 0;
    fclose(f);
    assert(strstr(buf, "stage,C,runnable,B_eff") != NULL);
    assert(strstr(buf, "v2,talker,8,3,3,0,int8,matmat,matmat_bf16,vnni,ragged,1") != NULL);
    return 0;
}
