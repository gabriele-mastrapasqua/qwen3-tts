#include "qwen_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void expect_text(const char *label, const char *json, const char *want) {
    char *got = NULL;
    const char *why = NULL;
    int status = qwen_json_extract_string(json, "text", &got, &why);
    if (status != QWEN_JSON_STRING_VALID || !got || strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s: got %s\n", label, got ? got : "<NULL>");
        failures++;
    }
    free(got);
}

static void expect_reject(const char *label, const char *json) {
    char *got = NULL;
    const char *why = NULL;
    int status = qwen_json_extract_string(json, "text", &got, &why);
    if (status != QWEN_JSON_STRING_INVALID || got || !why) {
        fprintf(stderr, "FAIL %s: malformed string status=%d value=%s\n",
                label, status, got ? got : "<NULL>");
        failures++;
    }
    free(got);
}

int main(void) {
    expect_text("ASCII", "{\"text\":\"hello\"}", "hello");
    expect_text("raw UTF-8 Italian", "{\"text\":\"giovedì\"}", "giovedì");
    expect_text("escaped Italian", "{\"text\":\"gioved\\u00ec\"}", "giovedì");
    expect_text("accent set", "{\"text\":\"\\u00e8 \\u00e9 \\u00e0 \\u00f2 \\u00f9\"}",
                "è é à ò ù");
    expect_text("quote slash newline", "{\"text\":\"quote: \\\" slash: \\\\ line\\nnext\"}",
                "quote: \" slash: \\ line\nnext");
    expect_text("all short escapes", "{\"text\":\"a\\/b\\b\\f\\r\\t\"}",
                "a/b\b\f\r\t");
    expect_text("surrogate pair", "{\"text\":\"\\uD83D\\uDE00\"}", "\xF0\x9F\x98\x80");

    expect_reject("lone high surrogate", "{\"text\":\"\\uD83D\"}");
    expect_reject("lone low surrogate", "{\"text\":\"\\uDE00\"}");
    expect_reject("malformed unicode escape", "{\"text\":\"\\u12G4\"}");
    expect_reject("truncated unicode escape", "{\"text\":\"\\u12\"}");
    expect_reject("NUL escape", "{\"text\":\"before\\u0000after\"}");
    expect_reject("unknown escape", "{\"text\":\"\\q\"}");
    expect_reject("truncated escape", "{\"text\":\"abc\\\"}");

    char *raw = NULL, *escaped = NULL;
    const char *why = NULL;
    int raw_status = qwen_json_extract_string("{\"text\":\"giovedì\"}", "text",
                                               &raw, &why);
    int escaped_status = qwen_json_extract_string("{\"text\":\"gioved\\u00ec\"}", "text",
                                                   &escaped, &why);
    if (raw_status != QWEN_JSON_STRING_VALID || escaped_status != QWEN_JSON_STRING_VALID ||
        !raw || !escaped || strlen(raw) != strlen(escaped) ||
        memcmp(raw, escaped, strlen(raw) + 1) != 0) {
        fprintf(stderr, "FAIL raw/escaped UTF-8 equivalence\n");
        failures++;
    }
    free(raw);
    free(escaped);

    char escaped_json[128];
    qwen_json_escape(escaped_json, sizeof(escaped_json), "giovedì \"ok\"\n");
    if (strcmp(escaped_json, "giovedì \\\"ok\\\"\\n") != 0) {
        fprintf(stderr, "FAIL UTF-8 JSON escaping: %s\n", escaped_json);
        failures++;
    }

    char *missing = NULL;
    if (qwen_json_extract_string("{}", "text", &missing, &why) != QWEN_JSON_STRING_ABSENT || missing) {
        fprintf(stderr, "FAIL absent field status\n");
        failures++;
    }

    if (failures) return 1;
    puts("PASS: JSON string parser tests");
    return 0;
}
