/*
 * qwen_tts_safetensors.h - Safetensors file format reader (multi-shard support)
 * Standard HuggingFace safetensors format with memory-mapped I/O.
 */

#ifndef QWEN_TTS_SAFETENSORS_H
#define QWEN_TTS_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#define SAFETENSORS_MAX_TENSORS 1024
#define SAFETENSORS_MAX_SHARDS 8

typedef enum {
    DTYPE_F32 = 0,
    DTYPE_F16 = 1,
    DTYPE_BF16 = 2,
    DTYPE_I32 = 3,
    DTYPE_I64 = 4,
    DTYPE_BOOL = 5,
    DTYPE_UNKNOWN = -1
} safetensor_dtype_t;

typedef struct {
    char name[256];
    safetensor_dtype_t dtype;
    int ndim;
    int64_t shape[8];
    size_t data_offset;
    size_t data_size;
} safetensor_t;

typedef struct {
    char *path;
    void *data;
    size_t file_size;
    size_t header_size;
    char *header_json;
    int num_tensors;
    safetensor_t tensors[SAFETENSORS_MAX_TENSORS];
} safetensors_file_t;

/* Multi-shard wrapper */
typedef struct {
    safetensors_file_t *shards[SAFETENSORS_MAX_SHARDS];
    int num_shards;
} multi_safetensors_t;

/* Open a single safetensors file (memory-mapped) */
safetensors_file_t *safetensors_open(const char *path);
void safetensors_close(safetensors_file_t *sf);

/* Open model from directory (auto-detects single file or multi-shard) */
multi_safetensors_t *multi_safetensors_open(const char *model_dir);
void multi_safetensors_close(multi_safetensors_t *ms);

/* Find a tensor by name across all shards */
const safetensor_t *multi_safetensors_find(const multi_safetensors_t *ms,
                                            const char *name,
                                            safetensors_file_t **out_sf);

/* Get raw pointer to tensor data (within mmap'd region) */
const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t);

/* Get tensor data as float32 (allocates, caller must free) */
float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t);

/* Get direct pointer to bf16 data in mmap'd region (no copy) */
uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t);

int safetensor_is_bf16(const safetensor_t *t);
int64_t safetensor_numel(const safetensor_t *t);
void safetensor_print(const safetensor_t *t);
void safetensors_print_all(const safetensors_file_t *sf);

#endif /* QWEN_TTS_SAFETENSORS_H */
