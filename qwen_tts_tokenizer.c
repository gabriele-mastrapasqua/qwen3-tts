/* qwen_tts_tokenizer.c - GPT-2 style byte-level BPE tokenizer for Qwen3-TTS */
#include "qwen_tts_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdatomic.h>

typedef struct {
    char *key;
    int   key_len;
    int32_t value;
} ht_entry_t;

typedef struct {
    ht_entry_t *entries;
    int capacity;
    int count;
} hash_table_t;

static uint32_t ht_hash(const char *key, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)key[i];
        h *= 16777619u;
    }
    return h;
}

static void ht_init(hash_table_t *ht, int capacity) {
    ht->capacity = capacity;
    ht->count = 0;
    ht->entries = (ht_entry_t *)calloc(capacity, sizeof(ht_entry_t));
}

static void ht_free(hash_table_t *ht) {
    for (int i = 0; i < ht->capacity; i++) {
        free(ht->entries[i].key);
    }
    free(ht->entries);
    ht->entries = NULL;
    ht->capacity = ht->count = 0;
}

static bool ht_set(hash_table_t *ht, const char *key, int key_len, int32_t value) {
    if (ht->count * 3 >= ht->capacity * 2) return false;
    uint32_t idx = ht_hash(key, key_len) % (uint32_t)ht->capacity;
    for (;;) {
        ht_entry_t *e = &ht->entries[idx];
        if (!e->key) {
            e->key = (char *)malloc(key_len + 1);
            memcpy(e->key, key, key_len);
            e->key[key_len] = '\0';
            e->key_len = key_len;
            e->value = value;
            ht->count++;
            return true;
        }
        if (e->key_len == key_len && memcmp(e->key, key, key_len) == 0) {
            e->value = value;
            return true;
        }
        idx = (idx + 1) % (uint32_t)ht->capacity;
    }
}

static bool ht_get(const hash_table_t *ht, const char *key, int key_len, int32_t *out) {
    if (!ht->entries || ht->count == 0) return false;
    uint32_t idx = ht_hash(key, key_len) % (uint32_t)ht->capacity;
    for (;;) {
        const ht_entry_t *e = &ht->entries[idx];
        if (!e->key) return false;
        if (e->key_len == key_len && memcmp(e->key, key, key_len) == 0) {
            *out = e->value;
            return true;
        }
        idx = (idx + 1) % (uint32_t)ht->capacity;
    }
}

static uint32_t byte_to_unicode[256];
static uint8_t  unicode_to_byte[512];
static atomic_bool byte_tables_initialized = false;

static void init_byte_tables(void) {
    if (atomic_load_explicit(&byte_tables_initialized, memory_order_acquire)) return;
    memset(unicode_to_byte, 0, sizeof(unicode_to_byte));

    int good_bytes[256];
    int n_good = 0;
    for (int b = '!'; b <= '~'; b++) good_bytes[n_good++] = b;
    for (int b = 0xa1; b <= 0xac; b++) good_bytes[n_good++] = b;
    for (int b = 0xae; b <= 0xff; b++) good_bytes[n_good++] = b;

    bool is_good[256] = {false};
    for (int i = 0; i < n_good; i++) {
        int b = good_bytes[i];
        byte_to_unicode[b] = (uint32_t)b;
        is_good[b] = true;
    }

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        if (!is_good[b]) {
            byte_to_unicode[b] = 256 + extra;
            extra++;
        }
    }

    for (int b = 0; b < 256; b++) {
        uint32_t cp = byte_to_unicode[b];
        if (cp < 512) unicode_to_byte[cp] = (uint8_t)b;
    }

    atomic_store_explicit(&byte_tables_initialized, true, memory_order_release);
}

static int cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static int byte_to_gpt2_utf8(uint8_t b, char *out) {
    return cp_to_utf8(byte_to_unicode[b], out);
}

struct qwen_tokenizer {
    hash_table_t vocab;
    hash_table_t merges;
    char **id_to_token;
    int max_id;
    int vocab_count;
};

static char *parse_json_string(const char *json, int *pos) {
    if (json[*pos] != '"') return NULL;
    (*pos)++;

    int len = 0;
    int p = *pos;
    while (json[p] && json[p] != '"') {
        if (json[p] == '\\') {
            p++;
            if (json[p] == 'u') {
                p += 4;
                len += 4;
            } else {
                len += 1;
            }
            p++;
        } else {
            unsigned char c = (unsigned char)json[p];
            if (c < 0x80) { len++; p++; }
            else if (c < 0xE0) { len += 2; p += 2; }
            else if (c < 0xF0) { len += 3; p += 3; }
            else { len += 4; p += 4; }
        }
    }

    char *out = (char *)malloc(len + 1);
    if (!out) return NULL;

    int o = 0;
    while (json[*pos] && json[*pos] != '"') {
        if (json[*pos] == '\\') {
            (*pos)++;
            switch (json[*pos]) {
                case '"':  out[o++] = '"'; break;
                case '\\': out[o++] = '\\'; break;
                case '/':  out[o++] = '/'; break;
                case 'n':  out[o++] = '\n'; break;
                case 'r':  out[o++] = '\r'; break;
                case 't':  out[o++] = '\t'; break;
                case 'b':  out[o++] = '\b'; break;
                case 'f':  out[o++] = '\f'; break;
                case 'u': {
                    (*pos)++;
                    if (!json[*pos] || !json[*pos+1] || !json[*pos+2] || !json[*pos+3]) {
                        while (json[*pos]) (*pos)++;
                        break;
                    }
                    uint32_t cp = 0;
                    for (int i = 0; i < 4; i++) {
                        char c = json[*pos + i];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= c - '0';
                        else if (c >= 'a' && c <= 'f') cp |= 10 + c - 'a';
                        else if (c >= 'A' && c <= 'F') cp |= 10 + c - 'A';
                    }
                    *pos += 3;
                    if (cp >= 0xD800 && cp <= 0xDBFF && json[*pos + 1] == '\\' && json[*pos + 2] == 'u' &&
                        json[*pos+3] && json[*pos+4] && json[*pos+5] && json[*pos+6]) {
                        *pos += 3;
                        uint32_t lo = 0;
                        for (int i = 0; i < 4; i++) {
                            char c = json[*pos + i];
                            lo <<= 4;
                            if (c >= '0' && c <= '9') lo |= c - '0';
                            else if (c >= 'a' && c <= 'f') lo |= 10 + c - 'a';
                            else if (c >= 'A' && c <= 'F') lo |= 10 + c - 'A';
                        }
                        *pos += 3;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    o += cp_to_utf8(cp, out + o);
                    break;
                }
                default: out[o++] = json[*pos]; break;
            }
            (*pos)++;
        } else {
            out[o++] = json[*pos];
            (*pos)++;
        }
    }
    out[o] = '\0';
    if (json[*pos] == '"') (*pos)++;
    return out;
}

static void skip_ws(const char *json, int *pos) {
    while (json[*pos] == ' ' || json[*pos] == '\t' || json[*pos] == '\n' || json[*pos] == '\r')
        (*pos)++;
}

static int32_t parse_json_int(const char *json, int *pos) {
    int32_t val = 0;
    int sign = 1;
    if (json[*pos] == '-') { sign = -1; (*pos)++; }
    while (json[*pos] >= '0' && json[*pos] <= '9') {
        val = val * 10 + (json[*pos] - '0');
        (*pos)++;
    }
    return val * sign;
}

static bool load_vocab(qwen_tokenizer_t *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return false; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = (char *)malloc(fsize + 1);
    if (!json) { fclose(f); return false; }
    size_t rd = fread(json, 1, fsize, f);
    fclose(f);
    if (rd != (size_t)fsize) {
        fprintf(stderr, "tokenizer: short read on %s (%zu/%ld bytes)\n", path, rd, fsize);
        free(json);
        return false;
    }
    json[fsize] = '\0';

    int est_size = 0;
    for (long i = 0; i < fsize; i++) if (json[i] == ':') est_size++;
    if (est_size < 1000) est_size = 200000;

    ht_init(&tok->vocab, est_size * 3);

    int pos = 0;
    skip_ws(json, &pos);
    if (json[pos] != '{') { free(json); return false; }
    pos++;

    tok->max_id = 0;
    tok->vocab_count = 0;

    while (json[pos]) {
        skip_ws(json, &pos);
        if (json[pos] == '}') break;
        if (json[pos] == ',') { pos++; continue; }

        char *key = parse_json_string(json, &pos);
        if (!key) break;

        skip_ws(json, &pos);
        if (json[pos] != ':') { free(key); break; }
        pos++;
        skip_ws(json, &pos);

        int32_t id = parse_json_int(json, &pos);

        int key_len = (int)strlen(key);
        ht_set(&tok->vocab, key, key_len, id);

        if (id > tok->max_id) tok->max_id = id;
        tok->vocab_count++;

        free(key);
    }

    tok->id_to_token = (char **)calloc(tok->max_id + 1, sizeof(char *));
    if (tok->id_to_token) {
        for (int i = 0; i < tok->vocab.capacity; i++) {
            ht_entry_t *e = &tok->vocab.entries[i];
            if (e->key && e->value >= 0 && e->value <= tok->max_id) {
                tok->id_to_token[e->value] = e->key;
            }
        }
    }

    free(json);
    fprintf(stderr, "tokenizer: loaded %d vocab entries (max_id=%d)\n", tok->vocab_count, tok->max_id);
    return true;
}

static char *make_merge_key(const char *a, int a_len, const char *b, int b_len, int *out_len) {
    int len = a_len + 1 + b_len;
    char *key = (char *)malloc(len + 1);
    memcpy(key, a, a_len);
    key[a_len] = '\xff';
    memcpy(key + a_len + 1, b, b_len);
    key[len] = '\0';
    *out_len = len;
    return key;
}

static bool load_merges(qwen_tokenizer_t *tok, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "tokenizer: cannot open %s\n", path); return false; }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *data = (char *)malloc(fsize + 1);
    if (!data) { fclose(f); return false; }
    size_t rd = fread(data, 1, fsize, f);
    fclose(f);
    if (rd != (size_t)fsize) {
        fprintf(stderr, "tokenizer: short read on %s (%zu/%ld bytes)\n", path, rd, fsize);
        free(data);
        return false;
    }
    data[fsize] = '\0';

    int n_lines = 0;
    for (long i = 0; i < fsize; i++) if (data[i] == '\n') n_lines++;
    n_lines += 1;

    ht_init(&tok->merges, n_lines * 3);

    int rank = 0;
    char *line = data;
    while (*line) {
        char *eol = strchr(line, '\n');
        int line_len = eol ? (int)(eol - line) : (int)strlen(line);

        if (line_len > 0 && line[0] != '#') {
            char *sep = NULL;
            for (int i = 0; i < line_len; i++) {
                if (line[i] == ' ') { sep = line + i; break; }
            }
            if (sep) {
                int a_len = (int)(sep - line);
                int b_len = line_len - a_len - 1;
                while (b_len > 0 && line[a_len + 1 + b_len - 1] == '\r') b_len--;

                if (a_len > 0 && b_len > 0) {
                    int key_len;
                    char *key = make_merge_key(line, a_len, sep + 1, b_len, &key_len);
                    ht_set(&tok->merges, key, key_len, rank);
                    free(key);
                    rank++;
                }
            }
        }

        if (!eol) break;
        line = eol + 1;
    }

    free(data);
    fprintf(stderr, "tokenizer: loaded %d merge rules\n", rank);
    return true;
}

static inline bool is_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static inline bool is_utf8_start(unsigned char c) {
    return c >= 0xC0;
}

static inline int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

typedef struct {
    char **chunks;
    int count;
    int capacity;
} chunk_list_t;

static void chunk_add(chunk_list_t *cl, const char *s, int len) {
    if (len <= 0) return;
    if (cl->count >= cl->capacity) {
        cl->capacity = cl->capacity ? cl->capacity * 2 : 64;
        cl->chunks = (char **)realloc(cl->chunks, cl->capacity * sizeof(char *));
    }
    char *chunk = (char *)malloc(len + 1);
    memcpy(chunk, s, len);
    chunk[len] = '\0';
    cl->chunks[cl->count++] = chunk;
}

static char **pre_tokenize(const char *text, int *n_chunks) {
    chunk_list_t cl = {NULL, 0, 0};
    int len = (int)strlen(text);
    int i = 0;

    while (i < len) {
        unsigned char c = (unsigned char)text[i];

        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (c == ' ' && i + 1 < len) {
                unsigned char next = (unsigned char)text[i + 1];
                if (next != ' ' && next != '\t' && next != '\n' && next != '\r') {
                    int start = i;
                    i++;

                    unsigned char nc = (unsigned char)text[i];
                    if (is_letter(nc) || is_utf8_start(nc)) {
                        while (i < len) {
                            unsigned char cc = (unsigned char)text[i];
                            if (is_letter(cc)) { i++; }
                            else if (is_utf8_start(cc)) {
                                int clen = utf8_char_len(cc);
                                if (i + clen > len) break;
                                i += clen;
                            }
                            else break;
                        }
                    } else if (nc >= '0' && nc <= '9') {
                        int dcount = 0;
                        while (i < len && text[i] >= '0' && text[i] <= '9' && dcount < 3) {
                            i++; dcount++;
                        }
                    } else if (nc != ' ' && nc != '\t' && nc != '\n' && nc != '\r') {
                        while (i < len) {
                            unsigned char cc = (unsigned char)text[i];
                            if (is_letter(cc) || (cc >= '0' && cc <= '9') ||
                                cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r' ||
                                is_utf8_start(cc)) break;
                            i++;
                        }
                    }
                    chunk_add(&cl, text + start, i - start);
                    continue;
                }
            }
            int start = i;
            while (i < len) {
                unsigned char cc = (unsigned char)text[i];
                if (cc != ' ' && cc != '\t' && cc != '\n' && cc != '\r') break;
                i++;
            }
            chunk_add(&cl, text + start, i - start);
            continue;
        }

        if (c == '\'' && i + 1 < len) {
            unsigned char next = (unsigned char)text[i + 1];
            int clen = 0;
            if (next == 's' || next == 't' || next == 'm' || next == 'd' ||
                next == 'S' || next == 'T' || next == 'M' || next == 'D') {
                clen = 2;
            } else if (i + 2 < len) {
                char pair[2] = {(char)next, text[i + 2]};
                if ((pair[0] == 'r' && pair[1] == 'e') || (pair[0] == 'R' && pair[1] == 'E') ||
                    (pair[0] == 'v' && pair[1] == 'e') || (pair[0] == 'V' && pair[1] == 'E') ||
                    (pair[0] == 'l' && pair[1] == 'l') || (pair[0] == 'L' && pair[1] == 'L')) {
                    clen = 3;
                }
            }
            if (clen > 0) {
                chunk_add(&cl, text + i, clen);
                i += clen;
                continue;
            }
        }

        {
            bool next_is_letter = false;
            int lookahead = i;
            if (!is_letter(c) && !(c >= '0' && c <= '9') && c != '\n' && c != '\r') {
                if (i + 1 < len) {
                    unsigned char nc = (unsigned char)text[i + 1];
                    if (is_letter(nc) || is_utf8_start(nc)) {
                        next_is_letter = true;
                        lookahead = i;
                    }
                }
            }
            if (is_letter(c) || is_utf8_start(c)) {
                next_is_letter = true;
                lookahead = i;
            }

            if (next_is_letter) {
                int start = lookahead;
                if (!is_letter((unsigned char)text[lookahead]) && !is_utf8_start((unsigned char)text[lookahead]))
                    lookahead++;
                while (lookahead < len) {
                    unsigned char cc = (unsigned char)text[lookahead];
                    if (is_letter(cc)) { lookahead++; }
                    else if (is_utf8_start(cc)) {
                        int clen = utf8_char_len(cc);
                        if (lookahead + clen > len) break;
                        lookahead += clen;
                    }
                    else break;
                }
                chunk_add(&cl, text + start, lookahead - start);
                i = lookahead;
                continue;
            }
        }

        if (c >= '0' && c <= '9') {
            int start = i;
            int dcount = 0;
            while (i < len && text[i] >= '0' && text[i] <= '9' && dcount < 3) {
                i++; dcount++;
            }
            chunk_add(&cl, text + start, i - start);
            continue;
        }

        {
            int start = i;
            while (i < len) {
                unsigned char cc = (unsigned char)text[i];
                if (is_letter(cc) || (cc >= '0' && cc <= '9') ||
                    cc == ' ' || cc == '\t' || cc == '\n' || cc == '\r' ||
                    is_utf8_start(cc)) break;
                i++;
            }
            while (i < len && (text[i] == '\r' || text[i] == '\n')) i++;
            if (i == start) i++;
            chunk_add(&cl, text + start, i - start);
        }
    }

    *n_chunks = cl.count;
    return cl.chunks;
}

typedef struct {
    char **pieces;
    int   *piece_lens;
    int    count;
    int    capacity;
} bpe_word_t;

static void bpe_word_init(bpe_word_t *w) {
    w->pieces = NULL;
    w->piece_lens = NULL;
    w->count = 0;
    w->capacity = 0;
}

static void bpe_word_push(bpe_word_t *w, const char *s, int len) {
    if (w->count >= w->capacity) {
        w->capacity = w->capacity ? w->capacity * 2 : 16;
        w->pieces = (char **)realloc(w->pieces, w->capacity * sizeof(char *));
        w->piece_lens = (int *)realloc(w->piece_lens, w->capacity * sizeof(int));
    }
    w->pieces[w->count] = (char *)malloc(len + 1);
    memcpy(w->pieces[w->count], s, len);
    w->pieces[w->count][len] = '\0';
    w->piece_lens[w->count] = len;
    w->count++;
}

static void bpe_word_free(bpe_word_t *w) {
    for (int i = 0; i < w->count; i++) free(w->pieces[i]);
    free(w->pieces);
    free(w->piece_lens);
    w->pieces = NULL;
    w->piece_lens = NULL;
    w->count = w->capacity = 0;
}

static void apply_bpe(bpe_word_t *word, const hash_table_t *merges) {
    while (word->count >= 2) {
        int best_rank = INT32_MAX;
        int best_idx = -1;

        for (int i = 0; i < word->count - 1; i++) {
            int key_len;
            char *key = make_merge_key(word->pieces[i], word->piece_lens[i],
                                       word->pieces[i + 1], word->piece_lens[i + 1],
                                       &key_len);
            int32_t rank;
            if (ht_get(merges, key, key_len, &rank) && rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
            free(key);
        }

        if (best_idx < 0) break;

        int new_len = word->piece_lens[best_idx] + word->piece_lens[best_idx + 1];
        char *merged = (char *)malloc(new_len + 1);
        memcpy(merged, word->pieces[best_idx], word->piece_lens[best_idx]);
        memcpy(merged + word->piece_lens[best_idx], word->pieces[best_idx + 1],
               word->piece_lens[best_idx + 1]);
        merged[new_len] = '\0';

        free(word->pieces[best_idx]);
        free(word->pieces[best_idx + 1]);
        word->pieces[best_idx] = merged;
        word->piece_lens[best_idx] = new_len;

        for (int i = best_idx + 1; i < word->count - 1; i++) {
            word->pieces[i] = word->pieces[i + 1];
            word->piece_lens[i] = word->piece_lens[i + 1];
        }
        word->count--;
    }
}

qwen_tokenizer_t *qwen_tokenizer_load(const char *dir) {
    init_byte_tables();

    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)calloc(1, sizeof(qwen_tokenizer_t));
    if (!tok) return NULL;

    int dir_len = (int)strlen(dir);
    char *vocab_path = (char *)malloc(dir_len + 20);
    char *merges_path = (char *)malloc(dir_len + 20);
    sprintf(vocab_path, "%s/vocab.json", dir);
    sprintf(merges_path, "%s/merges.txt", dir);

    bool ok = true;
    if (!load_vocab(tok, vocab_path)) ok = false;
    if (ok && !load_merges(tok, merges_path)) ok = false;

    free(vocab_path);
    free(merges_path);

    if (!ok) {
        qwen_tokenizer_free(tok);
        return NULL;
    }
    return tok;
}

qwen_tokenizer_t *qwen_tokenizer_load_files(const char *vocab_path, const char *merges_path) {
    init_byte_tables();

    qwen_tokenizer_t *tok = (qwen_tokenizer_t *)calloc(1, sizeof(qwen_tokenizer_t));
    if (!tok) return NULL;

    if (!load_vocab(tok, vocab_path) || !load_merges(tok, merges_path)) {
        qwen_tokenizer_free(tok);
        return NULL;
    }
    return tok;
}

int32_t *qwen_tokenizer_encode(qwen_tokenizer_t *tok, const char *text, int *out_len) {
    if (!tok || !text) { *out_len = 0; return NULL; }

    int n_chunks = 0;
    char **chunks = pre_tokenize(text, &n_chunks);
    if (!chunks || n_chunks == 0) { *out_len = 0; return NULL; }

    int cap = n_chunks * 4 + 16;
    int n = 0;
    int32_t *ids = (int32_t *)malloc(cap * sizeof(int32_t));

    for (int ci = 0; ci < n_chunks; ci++) {
        const char *chunk = chunks[ci];
        int chunk_len = (int)strlen(chunk);

        bpe_word_t word;
        bpe_word_init(&word);

        for (int j = 0; j < chunk_len; j++) {
            char buf[4];
            int blen = byte_to_gpt2_utf8((uint8_t)chunk[j], buf);
            bpe_word_push(&word, buf, blen);
        }

        apply_bpe(&word, &tok->merges);

        for (int j = 0; j < word.count; j++) {
            int32_t id;
            if (ht_get(&tok->vocab, word.pieces[j], word.piece_lens[j], &id)) {
                if (n >= cap) {
                    cap *= 2;
                    ids = (int32_t *)realloc(ids, cap * sizeof(int32_t));
                }
                ids[n++] = id;
            } else {
                for (int k = 0; k < word.piece_lens[j]; k++) {
                    char single[4];
                    int slen = byte_to_gpt2_utf8((uint8_t)word.pieces[j][k], single);
                    single[slen] = '\0';
                    int32_t sid;
                    if (ht_get(&tok->vocab, single, slen, &sid)) {
                        if (n >= cap) { cap *= 2; ids = (int32_t *)realloc(ids, cap * sizeof(int32_t)); }
                        ids[n++] = sid;
                    }
                }
            }
        }

        bpe_word_free(&word);
        free(chunks[ci]);
    }
    free(chunks);

    *out_len = n;
    return ids;
}

int32_t *qwen_tokenizer_encode_with_special(qwen_tokenizer_t *tok, const char *text,
                                             int add_bos, int add_eos, int *out_len) {
    (void)add_bos; (void)add_eos;
    return qwen_tokenizer_encode(tok, text, out_len);
}

int32_t *qwen_tokenizer_encode_para(qwen_tokenizer_t *tok, const char *text, int *out_len) {
    static const struct { const char *tag; int32_t id; } PARA_TAGS[] = {
        {"[laugh]",151646},{"[sigh]",151647},{"[cough]",151648},{"[breath]",151649},
        {"[sniff]",151650},{"[sneeze]",151651},{"[yawn]",151652},{"[uhm]",151653},
        {"[cry]",151654},{"[surprise]",151655},{"[confirm]",151656},{"[question]",151657},
        {"[grumble]",151658},{"[shh]",151659},{"[gasp]",151660},{"[groan]",151661},
    };
    const int NTAGS = (int)(sizeof(PARA_TAGS)/sizeof(PARA_TAGS[0]));
    if (!tok || !text) { if (out_len) *out_len = 0; return NULL; }

    int cap = 256, n = 0;
    int32_t *ids = (int32_t *)malloc(cap * sizeof(int32_t));
    if (!ids) { if (out_len) *out_len = 0; return NULL; }
    #define PARA_PUSH(v) do { if (n >= cap) { cap *= 2; ids = (int32_t *)realloc(ids, cap * sizeof(int32_t)); } ids[n++] = (v); } while (0)

    const char *p = text;
    const char *seg = text;
    while (*p) {
        int matched = -1, mlen = 0;
        if (*p == '[') {
            for (int i = 0; i < NTAGS; i++) {
                int tl = (int)strlen(PARA_TAGS[i].tag);
                if (strncmp(p, PARA_TAGS[i].tag, (size_t)tl) == 0) { matched = i; mlen = tl; break; }
            }
        }
        if (matched >= 0) {
            if (p > seg) {
                size_t sl = (size_t)(p - seg);
                char *sub = (char *)malloc(sl + 1);
                memcpy(sub, seg, sl); sub[sl] = '\0';
                int subn = 0;
                int32_t *sub_ids = qwen_tokenizer_encode(tok, sub, &subn);
                for (int i = 0; i < subn; i++) PARA_PUSH(sub_ids[i]);
                free(sub_ids); free(sub);
            }
            PARA_PUSH(PARA_TAGS[matched].id);
            p += mlen; seg = p;
        } else {
            p++;
        }
    }
    if (p > seg) {
        int subn = 0;
        int32_t *sub_ids = qwen_tokenizer_encode(tok, seg, &subn);
        for (int i = 0; i < subn; i++) PARA_PUSH(sub_ids[i]);
        free(sub_ids);
    }
    #undef PARA_PUSH
    if (out_len) *out_len = n;
    return ids;
}

char *qwen_tokenizer_decode(qwen_tokenizer_t *tok, const int32_t *tokens,
                            int num_tokens, int *out_len) {
    if (!tok || !tokens || num_tokens <= 0 || !tok->id_to_token) {
        if (out_len) *out_len = 0;
        return NULL;
    }

    int total = 0;
    for (int i = 0; i < num_tokens; i++) {
        int32_t id = tokens[i];
        if (id >= 0 && id <= tok->max_id && tok->id_to_token[id]) {
            total += (int)strlen(tok->id_to_token[id]);
        }
    }

    char *gpt2_str = (char *)malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < num_tokens; i++) {
        int32_t id = tokens[i];
        if (id >= 0 && id <= tok->max_id && tok->id_to_token[id]) {
            const char *t = tok->id_to_token[id];
            int tlen = (int)strlen(t);
            memcpy(gpt2_str + pos, t, tlen);
            pos += tlen;
        }
    }
    gpt2_str[pos] = '\0';

    char *out = (char *)malloc(pos + 1);
    int o = 0;
    int j = 0;
    while (j < pos) {
        unsigned char c = (unsigned char)gpt2_str[j];
        uint32_t cp;
        int clen;
        if (c < 0x80) { cp = c; clen = 1; }
        else if (c < 0xE0) { cp = c & 0x1F; clen = 2; }
        else if (c < 0xF0) { cp = c & 0x0F; clen = 3; }
        else { cp = c & 0x07; clen = 4; }
        for (int k = 1; k < clen && j + k < pos; k++)
            cp = (cp << 6) | ((unsigned char)gpt2_str[j + k] & 0x3F);
        j += clen;

        if (cp < 512) {
            out[o++] = (char)unicode_to_byte[cp];
        } else {
            o += cp_to_utf8(cp, out + o);
        }
    }
    out[o] = '\0';

    free(gpt2_str);
    if (out_len) *out_len = o;
    return out;
}

size_t qwen_tokenizer_vocab_size(qwen_tokenizer_t *tok) {
    if (!tok) return 0;
    return (size_t)(tok->max_id + 1);
}

int32_t qwen_tokenizer_get_special_token(qwen_tokenizer_t *tok, const char *name) {
    if (!tok || !name) return -1;
    int32_t id;
    int name_len = (int)strlen(name);
    if (ht_get(&tok->vocab, name, name_len, &id)) return id;
    return -1;
}

void qwen_tokenizer_free(qwen_tokenizer_t *tok) {
    if (!tok) return;
    ht_free(&tok->vocab);
    ht_free(&tok->merges);
    free(tok->id_to_token);
    free(tok);
}
