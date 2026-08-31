/* qwen_tts_gguf.c — load Talker weights from a quantized GGUF file. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include "qwen_tts.h"
#include "qwen_tts_kleidi.h"
#include "qwen_tts_kernels.h"
#include "qwen_tts_kleidi.h"
#include "qwen_tts_q8repack.h"
#include "ingot/dtype.h"
#include "ingot/gguf.h"
#include "ingot/quant.h"

static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(x >> 16);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return (uint16_t)(x >> 16);
}

void q4_from_ggml_pub(const uint8_t *src, q4_0_block_t *dst, size_t nblocks);
static void q4_from_ggml(const uint8_t *src, q4_0_block_t *dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t *s = src + b * 18;
        memcpy(&dst[b].scale_f16, s, 2);
        const uint8_t *qs = s + 2;
        for (int i = 0; i < 16; i++) {
            int k0 = 2 * i, k1 = 2 * i + 1;
            uint8_t v0 = (k0 < 16) ? (uint8_t)(qs[k0] & 0x0F) : (uint8_t)(qs[k0 - 16] >> 4);
            uint8_t v1 = (k1 < 16) ? (uint8_t)(qs[k1] & 0x0F) : (uint8_t)(qs[k1 - 16] >> 4);
            dst[b].qs[i] = (uint8_t)(v0 | (v1 << 4));
        }
    }
}

static int gguf_quant_prefill(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("QWEN_GGUF_QUANT_PREFILL"); v = (e && e[0] == '1'); }
    return v;
}

int qwen_gguf_override_talker(qwen_tts_ctx_t *ctx, const char *path, int silent) {
    if (!ctx || !path) return -1;

    ingot_gguf *g = NULL;
    char err[256] = "";
    if (ingot_gguf_open(&g, path, err, sizeof err) != 0) {
        fprintf(stderr, "Error: GGUF open failed for %s: %s\n", path, err);
        return -1;
    }

    const int nl     = ctx->config.num_layers;
    const int h      = ctx->config.hidden_size;
    const int inter  = ctx->config.intermediate_size;
    const int q_dim  = ctx->config.num_heads    * ctx->config.head_dim;
    const int kv_dim = ctx->config.num_kv_heads * ctx->config.head_dim;

    const struct { const char *kind; int rows; int cols; size_t off; size_t q4off; int fuse; size_t pref_off; } slots[] = {
        { "attn_q",      q_dim,  h,     offsetof(qwen_talker_layer_t, wq_bf16),   offsetof(qwen_talker_layer_t, wq_q4), 0, offsetof(qwen_talker_layer_t, wq_bf16_pref) },
        { "attn_k",      kv_dim, h,     offsetof(qwen_talker_layer_t, wk_bf16),   offsetof(qwen_talker_layer_t, wk_q4), 0, offsetof(qwen_talker_layer_t, wk_bf16_pref) },
        { "attn_v",      kv_dim, h,     offsetof(qwen_talker_layer_t, wv_bf16),   offsetof(qwen_talker_layer_t, wv_q4), 0, offsetof(qwen_talker_layer_t, wv_bf16_pref) },
        { "attn_output", h,      q_dim, offsetof(qwen_talker_layer_t, wo_bf16),   offsetof(qwen_talker_layer_t, wo_q4), 0, offsetof(qwen_talker_layer_t, wo_bf16_pref) },
        { "ffn_gate",    inter,  h,     offsetof(qwen_talker_layer_t, gate_bf16), offsetof(qwen_talker_layer_t, gate_up_fused_q4), 1, 0 },
        { "ffn_up",      inter,  h,     offsetof(qwen_talker_layer_t, up_bf16),   offsetof(qwen_talker_layer_t, gate_up_fused_q4), 2, 0 },
        { "ffn_down",    h,      inter, offsetof(qwen_talker_layer_t, down_bf16), offsetof(qwen_talker_layer_t, down_q4), 0, offsetof(qwen_talker_layer_t, down_bf16_pref) },
    };
    const int nslots = (int)(sizeof slots / sizeof slots[0]);

    size_t max_elems = 0;
    for (int s = 0; s < nslots; s++) {
        size_t n = (size_t)slots[s].rows * (size_t)slots[s].cols;
        if (n > max_elems) max_elems = n;
    }
    float *f32 = (float *)malloc(max_elems * sizeof(float));
    if (!f32) {
        ingot_gguf_close(g);
        fprintf(stderr, "Error: GGUF scratch alloc failed\n");
        return -1;
    }

    int applied = 0, missing = 0, mismatched = 0, native_q4 = 0, kai_packed = 0, q8_packed = 0;
    size_t by_pref = 0, by_dequant = 0, by_blocks = 0;
    const char *first_type = NULL;

    for (int li = 0; li < nl; li++) {
        qwen_talker_layer_t *l = &ctx->layers[li];
        if (!gguf_quant_prefill() && !l->gate_up_fused_bf16_pref && l->gate_up_fused_bf16) {
            size_t n = (size_t)(2 * inter) * h;
            uint16_t *cp = (uint16_t *)aligned_malloc(n * sizeof(uint16_t));
            if (cp) {
                memcpy(cp, l->gate_up_fused_bf16, n * sizeof(uint16_t));
                l->gate_up_fused_bf16_pref = cp;
                qwen_track_override(ctx, cp);
            }
        }
        uint8_t *fused_ggml = NULL;
        uint8_t *fused_q8 = NULL;
        int fused_bpr = 0;
        for (int s = 0; s < nslots; s++) {
            char name[128];
            snprintf(name, sizeof name, "blk.%d.%s.weight", li, slots[s].kind);

            const ingot_tensor *t = ingot_gguf_find(g, name);
            if (!t) { missing++; continue; }

            uint64_t shape[INGOT_MAX_RANK] = {0};
            ingot_gguf_shape_row_major(t, shape);
            if ((int)shape[0] != slots[s].rows || (int)shape[1] != slots[s].cols) {
                if (mismatched < 3)
                    fprintf(stderr, "Warning: %s has shape [%llu,%llu], expected [%d,%d] - skipped\n",
                            name, (unsigned long long)shape[0], (unsigned long long)shape[1],
                            slots[s].rows, slots[s].cols);
                mismatched++;
                continue;
            }

            (void)fused_bpr;
            if (ingot_gguf_dequant(g, t, f32) != 0) { missing++; continue; }

            size_t n = (size_t)slots[s].rows * (size_t)slots[s].cols;
            by_pref += n * sizeof(uint16_t);

            const int q4_native = (t->type == INGOT_TYPE_Q4_0 &&
                                   (slots[s].cols % Q4_0_BLOCK_SIZE) == 0 &&
                                   ingot_gguf_data(g, t) != NULL);
            uint16_t *bf = NULL;
            if (!q4_native) {
                bf = (uint16_t *)aligned_malloc(n * sizeof(uint16_t));
                if (!bf) { fprintf(stderr, "Error: GGUF bf16 alloc failed at %s\n", name); break; }
                for (size_t i = 0; i < n; i++) bf[i] = f32_to_bf16_rne(f32[i]);
                by_dequant += n * sizeof(uint16_t);
            }

            uint16_t **field = (uint16_t **)((char *)l + slots[s].off);
            if (!gguf_quant_prefill()) {
                switch (slots[s].fuse) {
                    case 1: case 2: break;
                    default: {
                        const uint16_t **pref = (const uint16_t **)((char *)l + slots[s].pref_off);
                        if (!*pref) *pref = *field;
                        break;
                    }
                }
            }
            if (bf) { *field = bf; qwen_track_override(ctx, bf); }
            applied++;

            if (t->type == INGOT_TYPE_Q8_0 && (slots[s].cols % Q8_0_BLOCK_SIZE) == 0) {
                const uint8_t *raw = (const uint8_t *)ingot_gguf_data(g, t);
                int bpr8 = slots[s].cols / Q8_0_BLOCK_SIZE;
                if (raw && !slots[s].fuse) {
                    if (qwen_q8r_register(bf, (const q8_0_block_t *)raw, slots[s].rows, slots[s].cols))
                        q8_packed++;
                } else if (raw) {
                    if (!fused_q8)
                        fused_q8 = (uint8_t *)malloc((size_t)(2 * inter) * bpr8 * sizeof(q8_0_block_t));
                    if (fused_q8) {
                        int odd = (slots[s].fuse == 2);
                        for (int r = 0; r < slots[s].rows; r++)
                            memcpy(fused_q8 + (size_t)(2 * r + odd) * bpr8 * sizeof(q8_0_block_t),
                                   raw + (size_t)r * bpr8 * sizeof(q8_0_block_t),
                                   (size_t)bpr8 * sizeof(q8_0_block_t));
                        if (odd) {
                            if (qwen_q8r_register(l->gate_up_fused_bf16,
                                                  (const q8_0_block_t *)fused_q8,
                                                  2 * inter, slots[s].cols))
                                q8_packed++;
                            free(fused_q8); fused_q8 = NULL;
                        }
                    }
                }
            }
            if (t->type == INGOT_TYPE_Q4_0 && (slots[s].cols % Q4_0_BLOCK_SIZE) == 0) {
                const uint8_t *raw = (const uint8_t *)ingot_gguf_data(g, t);
                if (raw) {
                    int bpr = slots[s].cols / Q4_0_BLOCK_SIZE;
                    q4_0_block_t **q4f = (q4_0_block_t **)((char *)l + slots[s].q4off);
                    int dst_rows = slots[s].fuse ? 2 * inter : slots[s].rows;
                    if (!*q4f) {
                        *q4f = (q4_0_block_t *)aligned_malloc((size_t)dst_rows * bpr * sizeof(q4_0_block_t));
                        if (*q4f) { qwen_track_override(ctx, *q4f);
                                    by_blocks += (size_t)dst_rows * bpr * sizeof(q4_0_block_t); }
                    }
                    if (*q4f) {
                        if (!slots[s].fuse) {
                            q4_from_ggml(raw, *q4f, (size_t)slots[s].rows * bpr);
                        } else {
                            int odd = (slots[s].fuse == 2);
                            for (int r = 0; r < slots[s].rows; r++)
                                q4_from_ggml(raw + (size_t)r * bpr * 18,
                                             *q4f + (size_t)(2 * r + odd) * bpr, (size_t)bpr);
                        }
                        if (native_q4 == 0) {
                            double worst = 0.0;
                            size_t nb = (size_t)slots[s].rows * bpr;
                            const q4_0_block_t *chk = slots[s].fuse ? NULL : *q4f;
                            if (chk) {
                                for (size_t b = 0; b < nb; b++) {
                                    float sc = qwen_f16_to_f32(chk[b].scale_f16);
                                    for (int i = 0; i < 16; i++) {
                                        float a0 = (float)((chk[b].qs[i] & 0x0F) - 8) * sc;
                                        float a1 = (float)((chk[b].qs[i] >> 4)   - 8) * sc;
                                        double d0 = fabs(a0 - f32[b * 32 + 2 * i]);
                                        double d1 = fabs(a1 - f32[b * 32 + 2 * i + 1]);
                                        if (d0 > worst) worst = d0;
                                        if (d1 > worst) worst = d1;
                                    }
                                }
                                fprintf(stderr, "GGUF: q4 re-nibble check on %s: max |diff| = %.3g%s\n",
                                        name, worst, worst == 0.0 ? " (exact)" : "  <-- NOT EXACT");
                            }
                        }
                        if (!slots[s].fuse) {
                            if (qwen_kleidi_register_q4(*q4f, raw, slots[s].rows, slots[s].cols))
                                kai_packed++;
                        } else {
                            if (!fused_ggml) {
                                fused_ggml = (uint8_t *)malloc((size_t)(2 * inter) * bpr * 18);
                                if (fused_ggml) fused_bpr = bpr;
                            }
                            if (fused_ggml) {
                                int odd = (slots[s].fuse == 2);
                                for (int r = 0; r < slots[s].rows; r++)
                                    memcpy(fused_ggml + (size_t)(2 * r + odd) * bpr * 18,
                                           raw + (size_t)r * bpr * 18, (size_t)bpr * 18);
                                if (odd) {
                                    if (qwen_kleidi_register_q4(*q4f, fused_ggml, 2 * inter, slots[s].cols))
                                        kai_packed++;
                                    free(fused_ggml); fused_ggml = NULL; fused_bpr = 0;
                                }
                            }
                        }
                        native_q4++;
                    }
                }
            }
            if (!first_type) first_type = ingot_type_name(t->type);
        }
        free(fused_ggml);
        free(fused_q8);
    }
    free(f32);

    for (int li = 0; li < nl; li++) {
        qwen_talker_layer_t *l = &ctx->layers[li];
        if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
            size_t row_bytes = (size_t)h * sizeof(uint16_t);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,   l->gate_bf16 + (size_t)r*h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h, l->up_bf16   + (size_t)r*h, row_bytes);
            }
        }
    }

    snprintf(ctx->src.talker_linear, sizeof ctx->src.talker_linear, "GGUF %s (%s)",
             first_type ? first_type : "?", path);
    ctx->src.talker_n = applied;
    ctx->src.talker_eligible = nl * nslots;

    if (!silent) {
        fprintf(stderr, "GGUF: %d/%d Talker tensors from %s (block type: %s)\n",
                applied, nl * nslots, path, first_type ? first_type : "?");
        if (native_q4)
            fprintf(stderr, "GGUF: %d tensors kept as NATIVE Q4_0 blocks (re-nibbled) -> q4 kernels\n",
                    native_q4);
        else
            fprintf(stderr, "GGUF: dequantized to bf16 - this measures the VALUES, not the format\n");
        {
            int kn = 0, qn = 0; size_t kb = 0, qb = 0;
            qwen_kleidi_stats(&kn, &kb);
            qwen_q8r_stats(&qn, &qb);
            fprintf(stderr, "GGUF: live bytes per representation (Talker)\n");
            fprintf(stderr, "        prefill bf16 (original, mmap)  %7.1f MB   read by prefill\n", by_pref / 1e6);
            fprintf(stderr, "        dequant bf16 (heap)            %7.1f MB   read ONLY on fallback\n", by_dequant / 1e6);
            if (by_blocks)
                fprintf(stderr, "        our q4 blocks (heap)           %7.1f MB   read by our q4 kernels\n", by_blocks / 1e6);
            if (kb) fprintf(stderr, "        KleidiAI packed (heap)         %7.1f MB   read by KleidiAI\n", kb / 1e6);
            if (qb) fprintf(stderr, "        q8_0x4 repacked (heap)         %7.1f MB   read by the i8mm path\n", qb / 1e6);
            fprintf(stderr, "        -> %d representations alive per matrix\n",
                    2 + (by_blocks ? 1 : 0) + ((kb || qb) ? 1 : 0) - 1);
        }
        if (q8_packed) {
            int np = 0; size_t kb = 0;
            qwen_q8r_stats(&np, &kb);
            fprintf(stderr, "GGUF: %d matrices repacked 4-row for ARM i8mm (%.1f MB, per-block-32 scales kept)\n",
                    q8_packed, (double)kb / 1e6);
        }
        if (kai_packed) {
            int np = 0; size_t kb = 0;
            qwen_kleidi_stats(&np, &kb);
            fprintf(stderr, "GGUF: %d matrices pre-packed for KleidiAI (%.1f MB, i8mm GEMM + dotprod GEMV)\n",
                    kai_packed, (double)kb / 1e6);
            float mx = 0.f, rel = 0.f;
            if (ctx->layers[0].wq_q4 &&
                qwen_kleidi_selfcheck(ctx->layers[0].wq_q4, q_dim, h, &mx, &rel))
                fprintf(stderr, "GGUF: KleidiAI vs our q4 kernel on blk.0.attn_q: "
                                "max|diff| %.4g, relative RMS %.3f%%%s\n",
                        mx, 100.0 * rel, (rel < 0.02f) ? "" : "   <-- SUSPICIOUS");
        } else if (native_q4 && qwen_kleidi_supported()) {
            fprintf(stderr, "GGUF: KleidiAI available but nothing packed - check QWEN_NO_KLEIDI\n");
        }
        if (missing || mismatched)
            fprintf(stderr, "GGUF: %d missing, %d shape mismatch\n", missing, mismatched);
    }

    ingot_gguf_close(g);
    return applied;
}

int qwen_gguf_override_rest(qwen_tts_ctx_t *ctx, const char *path, int silent) {
    if (!ctx || !path) return -1;

    ingot_gguf *g = NULL;
    char err[256] = "";
    if (ingot_gguf_open(&g, path, err, sizeof err) != 0) {
        fprintf(stderr, "Error: GGUF open failed for %s: %s\n", path, err);
        return -1;
    }

    const int nl    = ctx->config.cp_num_layers;
    const int h     = ctx->config.cp_hidden_size;
    const int inter = ctx->config.cp_intermediate_size;
    const int qd    = ctx->config.cp_num_heads    * ctx->config.cp_head_dim;
    const int kvd   = ctx->config.cp_num_kv_heads * ctx->config.cp_head_dim;

    const struct { const char *kind; int rows; int cols; size_t off; size_t q4off; int fuse; } slots[] = {
        { "attn_q",      qd,    h,     offsetof(qwen_cp_layer_t, wq_bf16),   offsetof(qwen_cp_layer_t, wq_q4), 0 },
        { "attn_k",      kvd,   h,     offsetof(qwen_cp_layer_t, wk_bf16),   offsetof(qwen_cp_layer_t, wk_q4), 0 },
        { "attn_v",      kvd,   h,     offsetof(qwen_cp_layer_t, wv_bf16),   offsetof(qwen_cp_layer_t, wv_q4), 0 },
        { "attn_output", h,     qd,    offsetof(qwen_cp_layer_t, wo_bf16),   offsetof(qwen_cp_layer_t, wo_q4), 0 },
        { "ffn_gate",    inter, h,     offsetof(qwen_cp_layer_t, gate_bf16), offsetof(qwen_cp_layer_t, gate_up_fused_q4), 1 },
        { "ffn_up",      inter, h,     offsetof(qwen_cp_layer_t, up_bf16),   offsetof(qwen_cp_layer_t, gate_up_fused_q4), 2 },
        { "ffn_down",    h,     inter, offsetof(qwen_cp_layer_t, down_bf16), offsetof(qwen_cp_layer_t, down_q4), 0 },
    };
    const int nslots = (int)(sizeof slots / sizeof slots[0]);

    size_t max_elems = (size_t)ctx->config.codebook_size * (size_t)h;
    { size_t ch = (size_t)ctx->config.codec_vocab_size * (size_t)ctx->config.hidden_size;
      if (ch > max_elems) max_elems = ch;
      size_t tp = (size_t)ctx->config.hidden_size * (size_t)ctx->config.text_hidden_size;
      if (tp > max_elems) max_elems = tp; }
    for (int s = 0; s < nslots; s++) {
        size_t n = (size_t)slots[s].rows * (size_t)slots[s].cols;
        if (n > max_elems) max_elems = n;
    }
    float *f32 = (float *)malloc(max_elems * sizeof(float));
    if (!f32) { ingot_gguf_close(g); fprintf(stderr, "Error: GGUF scratch alloc failed\n"); return -1; }

    int applied = 0, missing = 0, native_q4 = 0, kai_packed = 0, q8_packed = 0;
    int cp_applied = 0, heads_applied = 0;
    const char *first_type = NULL;

    #define REST_ONE(NAME, ROWS, COLS, BF_FIELD, Q4_FIELD, FUSE_MODE, FUSE_ROWS, FUSE_BUF, FUSE_BF)      \
    do {                                                                                        \
        const ingot_tensor *t = ingot_gguf_find(g, (NAME));                                     \
        if (!t) { missing++; break; }                                                           \
        uint64_t shp[INGOT_MAX_RANK] = {0};                                                     \
        ingot_gguf_shape_row_major(t, shp);                                                     \
        if ((int)shp[0] != (ROWS) || (int)shp[1] != (COLS)) {                                   \
            fprintf(stderr, "Warning: %s shape [%llu,%llu] != [%d,%d] - skipped\n", (NAME),     \
                    (unsigned long long)shp[0], (unsigned long long)shp[1], (ROWS), (COLS));    \
            missing++; break;                                                                   \
        }                                                                                       \
        size_t n = (size_t)(ROWS) * (size_t)(COLS);                                             \
                  \
        const int q4n = (t->type == INGOT_TYPE_Q4_0 && (COLS) % Q4_0_BLOCK_SIZE == 0 &&    \
                         ingot_gguf_data(g, t) != NULL);                                   \
        uint16_t *bf = *(BF_FIELD);                                                         \
        if (!q4n) {                                                                         \
            if (ingot_gguf_dequant(g, t, f32) != 0) { missing++; break; }                    \
            bf = (uint16_t *)aligned_malloc(n * sizeof(uint16_t));                          \
            if (!bf) { missing++; break; }                                                   \
            for (size_t i = 0; i < n; i++) bf[i] = f32_to_bf16_rne(f32[i]);                  \
            *(BF_FIELD) = bf;                                                                \
            qwen_track_override(ctx, bf);                                                    \
        }                                                                                    \
        applied++;                                                                               \
        if (!first_type) first_type = ingot_type_name(t->type);                                  \
        if (t->type == INGOT_TYPE_Q8_0 && (COLS) % Q8_0_BLOCK_SIZE == 0) {                       \
            const uint8_t *raw8 = (const uint8_t *)ingot_gguf_data(g, t);                        \
            int bpr8 = (COLS) / Q8_0_BLOCK_SIZE;                                                  \
            if (raw8 && !(FUSE_MODE)) {                                                           \
                if (qwen_q8r_register(bf, (const q8_0_block_t *)raw8, (ROWS), (COLS))) q8_packed++; \
            } else if (raw8) {                                                                    \
                int odd8 = ((FUSE_MODE) == 2);                                                    \
                if (!*(FUSE_BUF))                                                                 \
                    *(FUSE_BUF) = (uint8_t *)malloc((size_t)(FUSE_ROWS) * bpr8 * sizeof(q8_0_block_t)); \
                if (*(FUSE_BUF)) {                                                                \
                    for (int r = 0; r < (ROWS); r++)                                               \
                        memcpy(*(FUSE_BUF) + (size_t)(2 * r + odd8) * bpr8 * sizeof(q8_0_block_t), \
                               raw8 + (size_t)r * bpr8 * sizeof(q8_0_block_t),                     \
                               (size_t)bpr8 * sizeof(q8_0_block_t));                               \
                    if (odd8) {                                                                    \
                        if (qwen_q8r_register((FUSE_BF),                                            \
                                              (const q8_0_block_t *)*(FUSE_BUF),                    \
                                              (FUSE_ROWS), (COLS))) q8_packed++;                    \
                        free(*(FUSE_BUF)); *(FUSE_BUF) = NULL;                                      \
                    }                                                                              \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
        if (t->type == INGOT_TYPE_Q4_0 && (COLS) % Q4_0_BLOCK_SIZE == 0) {                       \
            const uint8_t *raw = (const uint8_t *)ingot_gguf_data(g, t);                          \
            int bpr = (COLS) / Q4_0_BLOCK_SIZE;                                                   \
            if (raw) {                                                                            \
                q4_0_block_t **q4dst = (Q4_FIELD);                                                    \
                int drows = (FUSE_MODE) ? (FUSE_ROWS) : (ROWS);                                   \
                if (!*q4dst) {                                                                      \
                    *q4dst = (q4_0_block_t *)aligned_malloc((size_t)drows * bpr * sizeof(q4_0_block_t)); \
                    if (*q4dst) qwen_track_override(ctx, *q4dst);                                     \
                }                                                                                 \
                if (*q4dst) {                                                                       \
                    if (!(FUSE_MODE)) {                                                           \
                        q4_from_ggml(raw, *q4dst, (size_t)(ROWS) * bpr);                            \
                        if (qwen_kleidi_register_q4(*q4dst, raw, (ROWS), (COLS))) kai_packed++;     \
                    } else {                                                                      \
                        int odd = ((FUSE_MODE) == 2);                                             \
                        for (int r = 0; r < (ROWS); r++)                                          \
                            q4_from_ggml(raw + (size_t)r * bpr * 18,                              \
                                         *q4dst + (size_t)(2 * r + odd) * bpr, (size_t)bpr);        \
                        if (!*(FUSE_BUF))                                                         \
                            *(FUSE_BUF) = (uint8_t *)malloc((size_t)(FUSE_ROWS) * bpr * 18);      \
                        if (*(FUSE_BUF)) {                                                        \
                            for (int r = 0; r < (ROWS); r++)                                      \
                                memcpy(*(FUSE_BUF) + (size_t)(2 * r + odd) * bpr * 18,            \
                                       raw + (size_t)r * bpr * 18, (size_t)bpr * 18);             \
                            if (odd) {                                                            \
                                if (qwen_kleidi_register_q4(*q4dst, *(FUSE_BUF), (FUSE_ROWS), (COLS))) \
                                    kai_packed++;                                                 \
                                free(*(FUSE_BUF)); *(FUSE_BUF) = NULL;                            \
                            }                                                                     \
                        }                                                                         \
                    }                                                                             \
                    native_q4++;                                                                  \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    } while (0)

    for (int li = 0; li < nl; li++) {
        qwen_cp_layer_t *l = &ctx->cp_layers[li];
        uint8_t *fused_ggml = NULL;
        for (int s = 0; s < nslots; s++) {
            char name[128];
            snprintf(name, sizeof name, "cp.blk.%d.%s.weight", li, slots[s].kind);
            uint16_t **bff = (uint16_t **)((char *)l + slots[s].off);
            q4_0_block_t **q4f = (q4_0_block_t **)((char *)l + slots[s].q4off);
            int before = applied;
            REST_ONE(name, slots[s].rows, slots[s].cols, bff, q4f,
                     slots[s].fuse, 2 * inter, &fused_ggml, l->gate_up_fused_bf16);
            cp_applied += (applied - before);
        }
        free(fused_ggml);
        if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
            size_t row_bytes = (size_t)h * sizeof(uint16_t);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,   l->gate_bf16 + (size_t)r*h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h, l->up_bf16   + (size_t)r*h, row_bytes);
            }
        }
    }

    for (int i = 0; i < 15; i++) {
        char name[64];
        snprintf(name, sizeof name, "cp.lm_head.%d.weight", i);
        uint16_t **bff = &ctx->cp_lm_head_bf16[i];
        q4_0_block_t **q4f = &ctx->cp_lm_head_q4[i];
        uint8_t *unused = NULL;
        int before = applied;
        REST_ONE(name, ctx->config.codebook_size, h, bff, q4f, 0, 0, &unused, NULL);
        heads_applied += (applied - before);
    }
    {
        uint8_t *unused = NULL;
        REST_ONE("codec_head.weight", ctx->config.codec_vocab_size, ctx->config.hidden_size,
                 &ctx->codec_head_bf16, &ctx->codec_head_q4, 0, 0, &unused, NULL);
        REST_ONE("text_proj.fc1.weight", ctx->config.text_hidden_size, ctx->config.text_hidden_size,
                 &ctx->text_proj_fc1_bf16, &ctx->text_proj_fc1_q4, 0, 0, &unused, NULL);
        REST_ONE("text_proj.fc2.weight", ctx->config.hidden_size, ctx->config.text_hidden_size,
                 &ctx->text_proj_fc2_bf16, &ctx->text_proj_fc2_q4, 0, 0, &unused, NULL);
        if (ctx->cp_mtp_proj_bf16)
            REST_ONE("cp.mtp_proj.weight", h, ctx->cp_emb_dim,
                     &ctx->cp_mtp_proj_bf16, &ctx->cp_mtp_proj_q4, 0, 0, &unused, NULL);
    }
    #undef REST_ONE

    free(f32);

    snprintf(ctx->src.cp_linear, sizeof ctx->src.cp_linear, "GGUF %s (%s)",
             first_type ? first_type : "?", path);
    ctx->src.cp_n        = cp_applied;
    ctx->src.cp_eligible = nl * nslots;
    ctx->src.cp_heads_n  = heads_applied;
    snprintf(ctx->src.cp_heads, sizeof ctx->src.cp_heads, "GGUF %s (%s)",
             first_type ? first_type : "?", path);
    ctx->src.extras_n = applied - cp_applied - heads_applied;
    snprintf(ctx->src.extras, sizeof ctx->src.extras, "GGUF %s (%s)",
             first_type ? first_type : "?", path);

    if (!silent) {
        fprintf(stderr, "GGUF-rest: %d tensors from %s (block type: %s)\n",
                applied, path, first_type ? first_type : "?");
        if (native_q4)
            fprintf(stderr, "GGUF-rest: %d kept as NATIVE Q4_0, %d packed for KleidiAI\n",
                    native_q4, kai_packed);
        if (q8_packed)
            fprintf(stderr, "GGUF-rest: %d matrices repacked 4-row for ARM i8mm (per-block-32 scales kept)\n",
                    q8_packed);
        if (missing)
            fprintf(stderr, "GGUF-rest: %d tensors missing or shape-mismatched\n", missing);
    }

    ingot_gguf_close(g);
    return applied;
}

void q4_from_ggml_pub(const uint8_t *src, q4_0_block_t *dst, size_t nblocks) {
    q4_from_ggml(src, dst, nblocks);
}

void qwen_report_model_sources(qwen_tts_ctx_t *ctx, const char *model_dir) {
    if (!ctx) return;
    const int cp_elig = ctx->src.cp_eligible ? ctx->src.cp_eligible
                                             : ctx->config.cp_num_layers * 7;
    const char *st = "safetensors BF16";
    fprintf(stderr, "\nMODEL SOURCES\n");
    fprintf(stderr, "  talker linear:      %s\n",
            ctx->src.talker_n ? ctx->src.talker_linear
                              : (ctx->use_int8 ? "safetensors BF16 -> INT8 at load"
                                               : (ctx->use_int4 ? "safetensors BF16 -> Q4_0 at load" : st)));
    fprintf(stderr, "  talker embedding:   %s\n", st);
    fprintf(stderr, "  CP linear:          %s\n",
            ctx->src.cp_n ? ctx->src.cp_linear
                          : (ctx->use_int8 ? "safetensors BF16 -> INT8 at load"
                                           : (ctx->use_int4 ? "safetensors BF16 -> Q4_0 at load" : st)));
    fprintf(stderr, "  CP lm_head x15:     %s\n",
            ctx->src.cp_heads_n ? ctx->src.cp_heads
                                : (ctx->use_int8 ? "safetensors BF16 -> INT8 at load" : st));
    fprintf(stderr, "  CP mtp_proj:        %s\n",
            ctx->src.extras_n ? ctx->src.extras
                              : (ctx->use_int8 ? "safetensors BF16 -> INT8 at load" : st));
    fprintf(stderr, "  CP embedding x15:   %s\n", st);
    fprintf(stderr, "  speech decoder:     safetensors F32 (%s/speech_tokenizer)\n",
            model_dir ? model_dir : "?");
    {
        int nq = 0, ni = 0, nb = 0; size_t bq = 0, bi = 0, bb = 0;
        qwen_kleidi_stats_by_kind(&nq, &bq, &ni, &bi, &nb, &bb);
        if (ni || nb || nq)
            fprintf(stderr, "  kleidiai packed:    int8 %d mat / %.0f MB · bf16 %d mat / %.0f MB"
                            " · q4 %d mat / %.0f MB\n",
                    ni, bi / 1048576.0, nb, bb / 1048576.0, nq, bq / 1048576.0);
    }
    fprintf(stderr, "  ---\n");
    if (ctx->src.talker_n)
        fprintf(stderr, "  Talker: %d/%d eligible tensors from %s\n",
                ctx->src.talker_n, ctx->src.talker_eligible, ctx->src.talker_linear);
    if (ctx->src.cp_n)
        fprintf(stderr, "  CP:     %d/%d eligible tensors from %s\n",
                ctx->src.cp_n, cp_elig, ctx->src.cp_linear);
    int cp_missing = ctx->src.cp_n ? (cp_elig - ctx->src.cp_n) : 0;
    int head_missing = ctx->src.cp_heads_n ? (15 - ctx->src.cp_heads_n) : 0;
    if (ctx->src.cp_n || ctx->src.talker_n)
        fprintf(stderr, "  CP safetensors quantized fallback: %d layer tensors, %d lm_heads\n",
                cp_missing, head_missing);
    fflush(stderr);
}
