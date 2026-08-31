/* qwen_tts_q8repack.h — GGUF Q8_0 weights, repacked 4-row, run on ARM i8mm/dotprod. */
#ifndef QWEN_TTS_Q8REPACK_H
#define QWEN_TTS_Q8REPACK_H

#include <stddef.h>
#include <stdint.h>

#define Q8_0_BLOCK_SIZE 32

typedef struct {
    uint16_t d;
    int8_t   qs[Q8_0_BLOCK_SIZE];
} q8_0_block_t;

typedef struct {
    uint16_t d[4];
    int8_t   qs[128];
} q8_0x4_block_t;

int qwen_q8r_supported(void);
int qwen_q8r_enabled(void);

size_t qwen_q8r_packed_bytes(int rows, int cols);
int    qwen_q8r_repack(q8_0x4_block_t *dst, const q8_0_block_t *src, int rows, int cols);

int qwen_q8r_derepack(q8_0_block_t *dst, const q8_0x4_block_t *src, int rows, int cols);

int qwen_q8r_register(const void *key, const q8_0_block_t *src, int rows, int cols);

int qwen_q8r_matmul(float *Y, const void *key, const float *X, int rows, int cols, int B);

void qwen_q8r_stats(int *n_packed, size_t *bytes);

#endif
