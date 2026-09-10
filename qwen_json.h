/* qwen_json.h - small JSON string decoder used by the HTTP server */
#ifndef QWEN_JSON_H
#define QWEN_JSON_H

#include <stddef.h>

/* Return the character after a valid JSON string beginning at the opening quote.
 * On failure return NULL and, when non-NULL, set *why to a short diagnostic. */
const char *qwen_json_string_end(const char *quoted, const char **why);

/* Decode a JSON string beginning at its opening quote.  The returned buffer is
 * heap allocated and must be freed by the caller.  When end is non-NULL it is
 * set to the character after the closing quote. */
char *qwen_json_decode_string(const char *quoted, const char **end, const char **why);

typedef enum {
    QWEN_JSON_STRING_ABSENT = 0,
    QWEN_JSON_STRING_VALID = 1,
    QWEN_JSON_STRING_INVALID = -1
} qwen_json_string_status_t;

/* Find a named JSON string member and decode its value.  *out is NULL unless
 * the return value is QWEN_JSON_STRING_VALID.  A malformed present field is
 * distinct from an absent optional field. */
qwen_json_string_status_t qwen_json_extract_string(const char *json, const char *key,
                                                   char **out, const char **why);

/* Escape a UTF-8 C string for a JSON string value.  Existing UTF-8 bytes pass
 * through unchanged; syntax and control bytes are escaped. */
void qwen_json_escape(char *dst, size_t dstsz, const char *src);

#endif
