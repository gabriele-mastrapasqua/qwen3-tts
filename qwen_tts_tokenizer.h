/* qwen_tts_tokenizer.h - Qwen3-TTS BPE Tokenizer */
#ifndef QWEN_TTS_TOKENIZER_H
#define QWEN_TTS_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen_tokenizer qwen_tokenizer_t;

#define QWEN_TTS_BOS_TOKEN_ID 151672
#define QWEN_TTS_EOS_TOKEN_ID 151673
#define QWEN_CODEC_BOS_ID     2149
#define QWEN_CODEC_EOS_ID     2150
#define QWEN_CODEC_PAD_ID     2148

#define QWEN_TOKENIZER_OK              0
#define QWEN_TOKENIZER_ERR_MEMORY     -1
#define QWEN_TOKENIZER_ERR_FILE       -2
#define QWEN_TOKENIZER_ERR_PARSE      -3
#define QWEN_TOKENIZER_ERR_NOT_FOUND  -4
#define QWEN_TOKENIZER_ERR_INVALID    -5

qwen_tokenizer_t *qwen_tokenizer_load(const char *dir);

qwen_tokenizer_t *qwen_tokenizer_load_files(const char *vocab_path, const char *merges_path);

int32_t *qwen_tokenizer_encode(qwen_tokenizer_t *tok, const char *text, int *out_len);

int32_t *qwen_tokenizer_encode_para(qwen_tokenizer_t *tok, const char *text, int *out_len);

int32_t *qwen_tokenizer_encode_with_special(qwen_tokenizer_t *tok, const char *text,
                                             int add_bos, int add_eos, int *out_len);

char *qwen_tokenizer_decode(qwen_tokenizer_t *tok, const int32_t *tokens,
                            int num_tokens, int *out_len);

size_t qwen_tokenizer_vocab_size(qwen_tokenizer_t *tok);

int32_t qwen_tokenizer_get_special_token(qwen_tokenizer_t *tok, const char *name);

void qwen_tokenizer_free(qwen_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif
