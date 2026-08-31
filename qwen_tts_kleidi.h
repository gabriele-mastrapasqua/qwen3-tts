/* qwen_tts_kleidi.h — Arm KleidiAI micro-kernels for GGUF Q4_0 weights. */
#ifndef QWEN_TTS_KLEIDI_H
#define QWEN_TTS_KLEIDI_H

#include <stddef.h>
#include <stdint.h>

int qwen_kleidi_supported(void);

int qwen_kleidi_enabled(void);

int qwen_kleidi_register_q4(const void *key, const uint8_t *ggml_blocks, int rows, int cols);

int qwen_kleidi_matmul_q4(float *Y, const void *key, const float *X, int rows, int cols, int B);

int qwen_kleidi_selfcheck(const void *key, int rows, int cols, float *max_abs, float *rel);

void qwen_kleidi_stats(int *n_packed, size_t *bytes);

int qwen_kleidi_register_i8(const void *key, const int8_t *W, const float *scale,
                            int rows, int cols);
int qwen_kleidi_matmul_i8(float *Y, const void *key, const float *X, int rows, int cols, int B);
int qwen_kleidi_matmul_i8_native(float *dst, const void *key, const float *lhs,
                                 size_t lhs_stride, size_t dst_stride,
                                 int rows, int cols, int B);

enum { QWEN_KAI_COMP_TALKER = 0, QWEN_KAI_COMP_CP, QWEN_KAI_COMP_N };
enum { QWEN_KAI_FAM_QKV = 0, QWEN_KAI_FAM_O, QWEN_KAI_FAM_FFN,
       QWEN_KAI_FAM_HEADS, QWEN_KAI_FAM_OTHER, QWEN_KAI_FAM_N };

int qwen_kleidi_register_i8_fam(const void *key, const int8_t *W, const float *scale,
                                int rows, int cols, int comp, int fam);
int qwen_kleidi_register_bf16_fam(const void *key, const uint16_t *W,
                                  int rows, int cols, int comp, int fam);
int qwen_kleidi_prefill_enabled(void);

int qwen_kleidi_matmul_i8_qkv_native(float *dq, float *dk, float *dv,
                                     const void *keyq, const void *keyk, const void *keyv,
                                     const float *lhs, size_t lhs_stride,
                                     int in_dim, int q_dim, int kv_dim, int B);

int qwen_kleidi_matmul_i8_qkv(float *q, float *k, float *v,
                              const void *keyq, const void *keyk, const void *keyv,
                              const float *x, int in_dim, int q_dim, int kv_dim);

int qwen_kleidi_register_bf16(const void *key, const uint16_t *W, int rows, int cols);
int qwen_kleidi_matmul_bf16(float *Y, const void *key, const float *X, int rows, int cols, int B);
int qwen_kleidi_matmul_bf16_native(float *dst, const void *key, const float *lhs,
                                   size_t lhs_stride, size_t dst_stride,
                                   int rows, int cols, int B);

int qwen_kleidi_i8_enabled(void);
int qwen_kleidi_bf16_enabled(void);

void qwen_kleidi_stats_by_kind(int *n_q4, size_t *b_q4, int *n_i8, size_t *b_i8,
                               int *n_bf, size_t *b_bf);

#endif
