/* qwen_json.c - small standards-correct JSON string decoder */
#include "qwen_json.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_value(unsigned char c) {
    if (c >= '0' && c <= '9') return (int)(c - '0');
    if (c >= 'a' && c <= 'f') return (int)(c - 'a') + 10;
    if (c >= 'A' && c <= 'F') return (int)(c - 'A') + 10;
    return -1;
}

static int parse_hex4(const unsigned char *p, uint16_t *out) {
    uint16_t v = 0;
    for (int i = 0; i < 4; i++) {
        if (p[i] == 0) return 0;
        int h = hex_value(p[i]);
        if (h < 0) return 0;
        v = (uint16_t)((v << 4) | (unsigned)h);
    }
    *out = v;
    return 1;
}

/* Return the length of a valid raw UTF-8 sequence, or zero for invalid input.
 * The boundary checks reject overlong encodings, UTF-16 surrogate encodings,
 * code points above U+10FFFF and truncated sequences. */
static int utf8_length(const unsigned char *p) {
    unsigned char c = p[0];
    if (c < 0x80) return c == 0 ? 0 : 1;
    if (c >= 0xC2 && c <= 0xDF) {
        return (p[1] >= 0x80 && p[1] <= 0xBF) ? 2 : 0;
    }
    if (c == 0xE0) {
        return (p[1] >= 0xA0 && p[1] <= 0xBF &&
                p[2] >= 0x80 && p[2] <= 0xBF) ? 3 : 0;
    }
    if ((c >= 0xE1 && c <= 0xEC) || (c >= 0xEE && c <= 0xEF)) {
        return (p[1] >= 0x80 && p[1] <= 0xBF &&
                p[2] >= 0x80 && p[2] <= 0xBF) ? 3 : 0;
    }
    if (c == 0xED) {
        return (p[1] >= 0x80 && p[1] <= 0x9F &&
                p[2] >= 0x80 && p[2] <= 0xBF) ? 3 : 0;
    }
    if (c == 0xF0) {
        return (p[1] >= 0x90 && p[1] <= 0xBF &&
                p[2] >= 0x80 && p[2] <= 0xBF &&
                p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    if (c >= 0xF1 && c <= 0xF3) {
        return (p[1] >= 0x80 && p[1] <= 0xBF &&
                p[2] >= 0x80 && p[2] <= 0xBF &&
                p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    if (c == 0xF4) {
        return (p[1] >= 0x80 && p[1] <= 0x8F &&
                p[2] >= 0x80 && p[2] <= 0xBF &&
                p[3] >= 0x80 && p[3] <= 0xBF) ? 4 : 0;
    }
    return 0;
}

static const char *json_error(const char **why, const char *message) {
    if (why) *why = message;
    return NULL;
}

const char *qwen_json_string_end(const char *quoted, const char **why) {
    if (why) *why = NULL;
    if (!quoted || *quoted != '"') return json_error(why, "expected a string");

    const unsigned char *p = (const unsigned char *)quoted + 1;
    for (;;) {
        unsigned char c = *p;
        if (c == 0) return json_error(why, "unterminated string");
        if (c == '"') return (const char *)p + 1;
        if (c < 0x20) return json_error(why, "control character in string");
        if (c != '\\') {
            int n = utf8_length(p);
            if (!n) return json_error(why, "invalid UTF-8 in string");
            p += n;
            continue;
        }

        p++;
        switch (*p) {
            case '"': case '\\': case '/':
            case 'b': case 'f': case 'n': case 'r': case 't':
                p++;
                break;
            case 'u': {
                uint16_t u;
                p++;
                if (!parse_hex4(p, &u)) return json_error(why, "bad \\u escape");
                p += 4;
                if (u >= 0xD800 && u <= 0xDBFF) {
                    uint16_t lo;
                    if (p[0] != '\\' || p[1] != 'u' || !parse_hex4(p + 2, &lo))
                        return json_error(why, "high surrogate without low surrogate");
                    if (lo < 0xDC00 || lo > 0xDFFF)
                        return json_error(why, "invalid low surrogate");
                    p += 6;
                } else if (u >= 0xDC00 && u <= 0xDFFF) {
                    return json_error(why, "lone low surrogate");
                } else if (u == 0) {
                    return json_error(why, "NUL escape is not supported in a C string field");
                }
                break;
            }
            default:
                return json_error(why, "bad escape in string");
        }
    }
}

static int append_utf8(char **dst, size_t *left, uint32_t cp) {
    unsigned char *p = (unsigned char *)*dst;
    if (cp <= 0x7F) {
        if (*left < 1) return 0;
        *p++ = (unsigned char)cp;
    } else if (cp <= 0x7FF) {
        if (*left < 2) return 0;
        *p++ = (unsigned char)(0xC0 | (cp >> 6));
        *p++ = (unsigned char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        if (*left < 3) return 0;
        *p++ = (unsigned char)(0xE0 | (cp >> 12));
        *p++ = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        *p++ = (unsigned char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0x10FFFF) {
        if (*left < 4) return 0;
        *p++ = (unsigned char)(0xF0 | (cp >> 18));
        *p++ = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
        *p++ = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        *p++ = (unsigned char)(0x80 | (cp & 0x3F));
    } else {
        return 0;
    }
    *left -= (size_t)(p - (unsigned char *)*dst);
    *dst = (char *)p;
    return 1;
}

char *qwen_json_decode_string(const char *quoted, const char **end, const char **why) {
    const char *finish = qwen_json_string_end(quoted, why);
    if (!finish) return NULL;
    size_t encoded_len = (size_t)(finish - quoted - 2); /* exclude quotes */
    char *result = (char *)malloc(encoded_len + 1);
    if (!result) {
        if (why) *why = "out of memory";
        return NULL;
    }

    const unsigned char *p = (const unsigned char *)quoted + 1;
    char *out = result;
    size_t left = encoded_len;
    while (p < (const unsigned char *)finish - 1) {
        if (*p != '\\') {
            int n = utf8_length(p);
            memcpy(out, p, (size_t)n);
            out += n; left -= (size_t)n; p += n;
            continue;
        }
        p++;
        switch (*p++) {
            case '"': *out++ = '"'; left--; break;
            case '\\': *out++ = '\\'; left--; break;
            case '/': *out++ = '/'; left--; break;
            case 'b': *out++ = '\b'; left--; break;
            case 'f': *out++ = '\f'; left--; break;
            case 'n': *out++ = '\n'; left--; break;
            case 'r': *out++ = '\r'; left--; break;
            case 't': *out++ = '\t'; left--; break;
            case 'u': {
                uint16_t u, lo = 0;
                (void)parse_hex4(p, &u);
                p += 4;
                uint32_t cp = u;
                if (u >= 0xD800 && u <= 0xDBFF) {
                    p += 2; /* backslash + 'u' */
                    (void)parse_hex4(p, &lo);
                    p += 4;
                    cp = 0x10000u + (((uint32_t)u - 0xD800u) << 10) + (lo - 0xDC00u);
                }
                if (!append_utf8(&out, &left, cp)) {
                    free(result);
                    if (why) *why = "decoded string too large";
                    return NULL;
                }
                break;
            }
            default:
                free(result);
                if (why) *why = "bad escape in string";
                return NULL;
        }
    }
    *out = '\0';
    if (end) *end = finish;
    return result;
}

qwen_json_string_status_t qwen_json_extract_string(const char *json, const char *key,
                                                   char **out, const char **why) {
    char pattern[256];
    if (out) *out = NULL;
    if (why) *why = NULL;
    if (!json || !key || snprintf(pattern, sizeof(pattern), "\"%s\"", key) >= (int)sizeof(pattern))
        return QWEN_JSON_STRING_ABSENT;
    const char *p = strstr(json, pattern);
    if (!p) return QWEN_JSON_STRING_ABSENT;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') p++;
    if (*p != '"') {
        if (why) *why = "expected a string";
        return QWEN_JSON_STRING_INVALID;
    }
    char *decoded = qwen_json_decode_string(p, NULL, why);
    if (!decoded) return QWEN_JSON_STRING_INVALID;
    if (out) *out = decoded;
    else free(decoded);
    return QWEN_JSON_STRING_VALID;
}

void qwen_json_escape(char *dst, size_t dstsz, const char *src) {
    if (!dst || dstsz == 0) return;
    if (!src) src = "";
    size_t j = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p; p++) {
        char escaped[7];
        const char *copy = NULL;
        size_t need = 1;
        switch (*p) {
            case '"': copy = "\\\""; need = 2; break;
            case '\\': copy = "\\\\"; need = 2; break;
            case '\b': copy = "\\b"; need = 2; break;
            case '\f': copy = "\\f"; need = 2; break;
            case '\n': copy = "\\n"; need = 2; break;
            case '\r': copy = "\\r"; need = 2; break;
            case '\t': copy = "\\t"; need = 2; break;
            default:
                if (*p < 0x20) {
                    (void)snprintf(escaped, sizeof(escaped), "\\u%04x", (unsigned)*p);
                    copy = escaped; need = 6;
                } else {
                    copy = (const char *)p;
                }
                break;
        }
        if (j + need + 1 > dstsz) break;
        memcpy(dst + j, copy, need);
        j += need;
    }
    dst[j] = '\0';
}
