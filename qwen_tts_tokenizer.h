/* qwen_tts_tokenizer.h - Qwen3-TTS BPE Tokenizer
 * 
 * GPT-2 style byte-level BPE tokenizer for Qwen3-TTS models.
 * Supports vocab.json and merges.txt loading.
 */

#ifndef QWEN_TTS_TOKENIZER_H
#define QWEN_TTS_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tokenizer context */
typedef struct qwen_tokenizer qwen_tokenizer_t;

/* Special token IDs for Qwen3-TTS */
#define QWEN_TTS_BOS_TOKEN_ID 151672
#define QWEN_TTS_EOS_TOKEN_ID 151673
#define QWEN_CODEC_BOS_ID     2149
#define QWEN_CODEC_EOS_ID     2150
#define QWEN_CODEC_PAD_ID     2148

/* Error codes */
#define QWEN_TOKENIZER_OK              0
#define QWEN_TOKENIZER_ERR_MEMORY     -1
#define QWEN_TOKENIZER_ERR_FILE       -2
#define QWEN_TOKENIZER_ERR_PARSE      -3
#define QWEN_TOKENIZER_ERR_NOT_FOUND  -4
#define QWEN_TOKENIZER_ERR_INVALID    -5

/* Load tokenizer from directory containing vocab.json and merges.txt */
qwen_tokenizer_t *qwen_tokenizer_load(const char *dir);

/* Load tokenizer from explicit file paths */
qwen_tokenizer_t *qwen_tokenizer_load_files(const char *vocab_path, const char *merges_path);

/* Encode text to token IDs */
int32_t *qwen_tokenizer_encode(qwen_tokenizer_t *tok, const char *text, int *out_len);

/* Encode text with special tokens (BOS/EOS) */
int32_t *qwen_tokenizer_encode_with_special(qwen_tokenizer_t *tok, const char *text, 
                                             int add_bos, int add_eos, int *out_len);

/* Decode token IDs back to text */
char *qwen_tokenizer_decode(qwen_tokenizer_t *tok, const int32_t *tokens, 
                            int num_tokens, int *out_len);

/* Get vocabulary size */
size_t qwen_tokenizer_vocab_size(qwen_tokenizer_t *tok);

/* Get token ID for a special token name */
int32_t qwen_tokenizer_get_special_token(qwen_tokenizer_t *tok, const char *name);

/* Free tokenizer and all allocated memory */
void qwen_tokenizer_free(qwen_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TTS_TOKENIZER_H */
