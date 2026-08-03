/* Shared internals. Not installed, not part of the public API.
 * SPDX-License-Identifier: MIT */
#ifndef INGOT_INTERNAL_H
#define INGOT_INTERNAL_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── errors ─────────────────────────────────────────────────────────────── */
static inline void ingot_err(char *err, size_t errsz, const char *fmt, ...) {
    if (err == NULL || errsz == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, errsz, fmt, args);
    va_end(args);
}

/* ── overflow-checked arithmetic ────────────────────────────────────────── */
static inline int ingot_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b > UINT64_MAX - a) return -1;
    *out = a + b;
    return 0;
}
static inline int ingot_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

/* ── bulk dtype conversions (dtype.c) ───────────────────────────────────────
 * The vectorized bodies behind ingot_dtype_to_f32 and the F16/BF16 cases of
 * ingot_dequant. `p` is the little-endian file bytes, not necessarily aligned.
 * Byte-for-byte identical to the per-element scalar functions — widening
 * float conversions are exact, and the tests hold them to that. */
void ingot_bf16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f16_block_to_f32(const unsigned char *p, size_t nelem, float *dst);
void ingot_f32_block_to_bf16(const float *src, size_t nelem, unsigned char *dst);

/* ── little-endian loads ────────────────────────────────────────────────────
 * Explicit, byte by byte: both container formats are defined little-endian and
 * a memcpy would silently do the wrong thing on a big-endian host. */
static inline uint16_t ingot_ld_u16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static inline uint32_t ingot_ld_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t ingot_ld_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

/* ── name index ─────────────────────────────────────────────────────────────
 * FNV-1a plus open addressing over a power-of-two table, slots holding
 * index+1 so that 0 means empty. Turns tensor lookup from the O(n) strcmp
 * walk every source project shipped into O(1) — it matters: a 400-tensor
 * model looked up 400 times is 160k strcmp for nothing. */
typedef struct {
    size_t   *slots;
    uint64_t *hashes;    /* parallel to the caller's item array */
    size_t    capacity;  /* power of two, 0 when unbuilt */
} ingot_index;

static inline uint64_t ingot_hash(const char *name, size_t len) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)name[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}
static inline uint64_t ingot_hash_str(const char *name) {
    return ingot_hash(name, strlen(name));
}

/* Build an index over `count` items; `name_of(items, i)` yields each name. */
typedef const char *(*ingot_name_fn)(const void *items, size_t i);

static inline int ingot_index_build(ingot_index *ix, const void *items,
                                    size_t count, ingot_name_fn name_of) {
    memset(ix, 0, sizeof(*ix));
    if (count == 0) return 0;
    if (count > SIZE_MAX / 4) return -1;
    size_t capacity = 16;
    while (capacity < count * 2) capacity *= 2;
    ix->slots  = (size_t *)calloc(capacity, sizeof(*ix->slots));
    ix->hashes = (uint64_t *)calloc(count, sizeof(*ix->hashes));
    if (ix->slots == NULL || ix->hashes == NULL) {
        free(ix->slots); free(ix->hashes);
        memset(ix, 0, sizeof(*ix));
        return -1;
    }
    ix->capacity = capacity;
    const size_t mask = capacity - 1;
    for (size_t i = 0; i < count; i++) {
        ix->hashes[i] = ingot_hash_str(name_of(items, i));
        size_t slot = (size_t)ix->hashes[i] & mask;
        while (ix->slots[slot] != 0) slot = (slot + 1) & mask;
        ix->slots[slot] = i + 1;
    }
    return 0;
}

/* Returns the item index, or (size_t)-1 when absent. */
static inline size_t ingot_index_find(const ingot_index *ix, const void *items,
                                      ingot_name_fn name_of, const char *name) {
    if (ix->capacity == 0 || name == NULL) return (size_t)-1;
    const uint64_t h = ingot_hash_str(name);
    const size_t mask = ix->capacity - 1;
    size_t slot = (size_t)h & mask;
    for (;;) {
        const size_t stored = ix->slots[slot];
        if (stored == 0) return (size_t)-1;
        const size_t i = stored - 1;
        if (ix->hashes[i] == h && strcmp(name_of(items, i), name) == 0) return i;
        slot = (slot + 1) & mask;
    }
}

static inline void ingot_index_free(ingot_index *ix) {
    free(ix->slots);
    free(ix->hashes);
    memset(ix, 0, sizeof(*ix));
}

/* ── misc ───────────────────────────────────────────────────────────────── */
static inline char *ingot_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *copy = (char *)malloc(n);
    if (copy != NULL) memcpy(copy, s, n);
    return copy;
}

static inline char *ingot_strndup(const char *s, size_t n) {
    char *copy = (char *)malloc(n + 1);
    if (copy != NULL) { memcpy(copy, s, n); copy[n] = '\0'; }
    return copy;
}

#endif
