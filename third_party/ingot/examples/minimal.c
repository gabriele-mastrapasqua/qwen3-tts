/* The whole library, used the way a small engine would use it.
 *
 * Build it against the two-file drop-in — no include path, no Makefile change:
 *
 *     cc -std=c11 -O2 -I../amalgam minimal.c ../amalgam/ingot.c -lpthread -lm
 *
 * or against the library:
 *
 *     cc -std=c11 -O2 -I../include minimal.c ../libingot.a -lpthread -lm
 *
 * Run it on anything: a .gguf, a .safetensors, or a model directory.
 *
 * SPDX-License-Identifier: MIT */
#include <stdio.h>
#include <stdlib.h>

#include <ingot/wfile.h>
#include <ingot/quant.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf | model.safetensors | model-dir>\n", argv[0]);
        return 2;
    }

    /* One handle for either container: the magic decides, not the caller. */
    char err[256];
    ingot_wfile *w = NULL;
    if (ingot_wfile_open(&w, argv[1], err, sizeof err) != 0) {
        fprintf(stderr, "%s\n", err);        /* the library never prints */
        return 1;
    }
    printf("%s: %zu tensors\n",
           ingot_wfile_container(w) == INGOT_CONTAINER_GGUF ? "GGUF" : "safetensors",
           ingot_wfile_count(w));

    /* Find the widest 2D tensor and multiply a vector through it, whatever
     * format it happens to be stored in. No branch on the type: that is what
     * the generic entry points are for. */
    const ingot_wtensor *best = NULL;
    for (size_t i = 0; i < ingot_wfile_count(w); i++) {
        const ingot_wtensor *t = ingot_wfile_at(w, i);
        if (t->rank != 2) continue;
        if (best == NULL || t->nelem > best->nelem) best = t;
    }
    if (best == NULL) {
        printf("no 2D tensor to multiply\n");
        ingot_wfile_close(w);
        return 0;
    }

    const size_t rows = (size_t)best->shape[0];
    const size_t cols = (size_t)best->shape[1];
    const char *format = best->ggml_type >= 0 ? ingot_type_name(best->ggml_type)
                                              : ingot_dtype_name(best->dtype);
    printf("largest: %s  %s  [%zu, %zu]  %s\n", best->name, format, rows, cols,
           best->ggml_type >= 0 && ingot_has_kernel(best->ggml_type)
               ? "(has a hand-written kernel)" : "(generic path)");

    float *x = (float *)calloc(cols, sizeof(float));
    float *y = (float *)calloc(rows, sizeof(float));
    if (x == NULL || y == NULL) { free(x); free(y); ingot_wfile_close(w); return 1; }
    for (size_t i = 0; i < cols; i++) x[i] = 1.0f / (float)cols;

    int rc;
    if (best->ggml_type >= 0) {
        /* GGUF: the ggml type multiplies in place, no dequantization. */
        rc = ingot_matvec(best->ggml_type, best->data, rows, cols, x, y);
    } else if (best->dtype == INGOT_DT_F32) {
        rc = ingot_matvec(INGOT_TYPE_F32, best->data, rows, cols, x, y);
    } else {
        /* safetensors dtypes other than f32 need one conversion first. */
        float *f32 = (float *)malloc((size_t)best->nelem * sizeof(float));
        rc = f32 != NULL && ingot_wfile_to_f32(w, best, f32) == 0
                 ? ingot_matvec(INGOT_TYPE_F32, f32, rows, cols, x, y) : -1;
        free(f32);
    }

    if (rc != 0) printf("that format has no decode in this build\n");
    else printf("mean of the row means: %g\n", (double)y[0]);

    free(x);
    free(y);
    ingot_wfile_close(w);
    return 0;
}
