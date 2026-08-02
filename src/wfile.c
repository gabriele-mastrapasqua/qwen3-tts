/* One handle for either container. See ingot/wfile.h for why this layer is
 * thin on purpose.
 *
 * SPDX-License-Identifier: MIT */
#include "ingot/wfile.h"
#include "internal.h"

#include <stdio.h>

#ifndef INGOT_NO_KERNELS
#include "ingot/quant.h"
#endif

struct ingot_wfile {
    ingot_container container;
    ingot_gguf     *gguf;
    ingot_st       *st;
    ingot_wtensor  *tensors;
    size_t          ntensor;
    ingot_index     index;
};

static const char *w_name_of(const void *items, size_t i) {
    return ((const ingot_wtensor *)items)[i].name;
}

static int build_from_gguf(ingot_wfile *w, char *err, size_t errsz) {
    w->ntensor = ingot_gguf_count(w->gguf);
    w->tensors = (ingot_wtensor *)calloc(w->ntensor ? w->ntensor : 1, sizeof(*w->tensors));
    if (w->tensors == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    for (size_t i = 0; i < w->ntensor; i++) {
        const ingot_tensor *t = ingot_gguf_at(w->gguf, i);
        ingot_wtensor *o = &w->tensors[i];
        o->name = t->name;
        o->ggml_type = t->type;
        o->rank = t->rank;
        /* ggml stores ne[0] as the fastest dimension; everything outside ggml
         * reads shapes the other way round, so this is where the flip lives —
         * once, explicitly, instead of in every consumer. */
        ingot_gguf_shape_row_major(t, o->shape);
        o->nelem = t->nelem;
        o->nbytes = t->nbytes;
        o->data = ingot_gguf_data(w->gguf, t);
        switch (t->type) {
        case INGOT_TYPE_F32:  o->dtype = INGOT_DT_F32;  break;
        case INGOT_TYPE_F16:  o->dtype = INGOT_DT_F16;  break;
        case INGOT_TYPE_BF16: o->dtype = INGOT_DT_BF16; break;
        case INGOT_TYPE_F64:  o->dtype = INGOT_DT_F64;  break;
        case INGOT_TYPE_I8:   o->dtype = INGOT_DT_I8;   break;
        case INGOT_TYPE_I16:  o->dtype = INGOT_DT_I16;  break;
        case INGOT_TYPE_I32:  o->dtype = INGOT_DT_I32;  break;
        case INGOT_TYPE_I64:  o->dtype = INGOT_DT_I64;  break;
        default:              o->dtype = INGOT_DT_UNKNOWN; break;  /* a block type */
        }
        if (o->data == NULL) {
            ingot_err(err, errsz, "tensor '%s' has no readable payload", t->name);
            return -1;
        }
    }
    return 0;
}

static int build_from_st(ingot_wfile *w, char *err, size_t errsz) {
    w->ntensor = ingot_st_count(w->st);
    w->tensors = (ingot_wtensor *)calloc(w->ntensor ? w->ntensor : 1, sizeof(*w->tensors));
    if (w->tensors == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }
    for (size_t i = 0; i < w->ntensor; i++) {
        const ingot_st_tensor *t = ingot_st_at(w->st, i);
        ingot_wtensor *o = &w->tensors[i];
        o->name = t->name;
        o->dtype = t->dtype;
        o->ggml_type = -1;
        o->rank = t->rank;
        for (uint32_t d = 0; d < t->rank; d++) o->shape[d] = t->shape[d];
        o->nelem = t->nelem;
        o->nbytes = t->nbytes;
        o->data = ingot_st_data(w->st, t);
        if (o->data == NULL) {
            ingot_err(err, errsz, "tensor '%s' has no readable payload", t->name);
            return -1;
        }
    }
    return 0;
}

int ingot_wfile_open(ingot_wfile **out, const char *path, char *err, size_t errsz) {
    if (out == NULL || path == NULL) {
        ingot_err(err, errsz, "ingot_wfile_open: null argument");
        return -1;
    }
    *out = NULL;
    unsigned char magic[4] = {0};
    FILE *probe = fopen(path, "rb");
    if (probe != NULL) {
        if (fread(magic, 1, 4, probe) != 4) memset(magic, 0, sizeof magic);
        fclose(probe);
    }
    ingot_wfile *w = (ingot_wfile *)calloc(1, sizeof(*w));
    if (w == NULL) { ingot_err(err, errsz, "out of memory"); return -1; }

    if (memcmp(magic, "GGUF", 4) == 0) {
        w->container = INGOT_CONTAINER_GGUF;
        if (ingot_gguf_open_split(&w->gguf, path, err, errsz) != 0) goto fail;
        if (build_from_gguf(w, err, errsz) != 0) goto fail;
    } else {
        w->container = INGOT_CONTAINER_SAFETENSORS;
        if (ingot_st_open(&w->st, path, err, errsz) != 0) goto fail;
        if (build_from_st(w, err, errsz) != 0) goto fail;
    }
    if (ingot_index_build(&w->index, w->tensors, w->ntensor, w_name_of) != 0) {
        ingot_err(err, errsz, "out of memory building the name index");
        goto fail;
    }
    *out = w;
    return 0;
fail:
    ingot_wfile_close(w);
    return -1;
}

void ingot_wfile_close(ingot_wfile *w) {
    if (w == NULL) return;
    ingot_index_free(&w->index);
    free(w->tensors);
    ingot_gguf_close(w->gguf);
    ingot_st_close(w->st);
    free(w);
}

ingot_container ingot_wfile_container(const ingot_wfile *w) {
    return w != NULL ? w->container : INGOT_CONTAINER_GGUF;
}
size_t ingot_wfile_count(const ingot_wfile *w) { return w != NULL ? w->ntensor : 0; }

const ingot_wtensor *ingot_wfile_at(const ingot_wfile *w, size_t index) {
    return (w != NULL && index < w->ntensor) ? &w->tensors[index] : NULL;
}

const ingot_wtensor *ingot_wfile_find(const ingot_wfile *w, const char *name) {
    if (w == NULL) return NULL;
    const size_t i = ingot_index_find(&w->index, w->tensors, w_name_of, name);
    return i == (size_t)-1 ? NULL : &w->tensors[i];
}

int ingot_wfile_to_f32(const ingot_wfile *w, const ingot_wtensor *t, float *dst) {
    if (w == NULL || t == NULL || t->data == NULL || dst == NULL) return -1;
    if (t->nelem > SIZE_MAX) return -1;
    if (t->ggml_type >= 0) {
#ifdef INGOT_NO_KERNELS
        (void)dst;
        return -1;
#else
        return ingot_dequant(t->ggml_type, t->data, (size_t)t->nelem, dst);
#endif
    }
    return ingot_dtype_to_f32(t->dtype, t->data, (size_t)t->nelem, dst);
}

const ingot_gguf *ingot_wfile_gguf(const ingot_wfile *w) {
    return (w != NULL && w->container == INGOT_CONTAINER_GGUF) ? w->gguf : NULL;
}
ingot_st *ingot_wfile_st(const ingot_wfile *w) {
    return (w != NULL && w->container == INGOT_CONTAINER_SAFETENSORS) ? w->st : NULL;
}
