/* qwen_tts_q4export.c — write a GGUF Q4_0 quantized by OUR weighted-LSQ quantizer.
 *
 * WHY
 * `llama-quantize`'s Q4_0 is plain absmax RTN. Ours (qwen_quantize_bf16_to_q4_0) is a
 * weighted least-squares fit. On this customer finetune the difference is not academic:
 * measured 2026-08-22, same speaker, same seed, same text —
 *     our 4-bit (LSQ)          94.6 % language identity,  0.4 % English
 *     GGUF Q4_0 (llama.cpp RTN) 0.0 % language identity, 96.8 % English   (same kernel!)
 * while their AVERAGE weight error differs by 0.03 percentage points (8.786 vs 8.820).
 * The finetune's delta from base is at most ~0.002 per weight and the 4-bit step is
 * ~absmax/8 ≈ 0.006: the finetune lives BELOW the quantization step, so how the scale
 * is chosen decides whether it survives at all.
 *
 * WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
 * It runs the SAME function that produces `--int4`, on the same bf16 weights, and only
 * changes the SERIALIZATION: our q4_0_block_t pairs nibbles adjacently, ggml's
 * block_q4_0 pairs k with k+16. Nothing is dequantized and re-quantized on the way —
 * that would be a different quantizer with the same name.
 *
 * The written file is a drop-in for both --gguf-talker and --gguf-rest: the two name
 * spaces (`blk.*` and `cp.*`/`codec_head`/`text_proj.*`) do not collide.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qwen_tts.h"
#include "qwen_tts_kernels.h"
#include "ingot/dtype.h"
#include "ingot/gguf.h"
#include "ingot/write.h"

/* our adjacent-pair nibbles -> ggml's split-half. Exact inverse of q4_from_ggml(). */
static void q4_to_ggml(uint8_t *dst, const q4_0_block_t *src, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        uint8_t *o = dst + b * 18;
        memcpy(o, &src[b].scale_f16, 2);
        const uint8_t *qs = src[b].qs;
        for (int k = 0; k < 16; k++) {
            /* value at index k goes in the low nibble, k+16 in the high nibble */
            uint8_t lo = (k % 2 == 0) ? (uint8_t)(qs[k / 2] & 0x0F) : (uint8_t)(qs[k / 2] >> 4);
            int k2 = k + 16;
            uint8_t hi = (k2 % 2 == 0) ? (uint8_t)(qs[k2 / 2] & 0x0F) : (uint8_t)(qs[k2 / 2] >> 4);
            o[2 + k] = (uint8_t)(lo | (hi << 4));
        }
    }
}

typedef struct { const char *name; const uint16_t *w; int rows, cols; } q4x_item_t;

/* Every tensor gets its OWN buffer, kept alive until save. `ingot_gguf_add_tensor`
 * stores the POINTER, it does not copy: reusing one scratch buffer made the writer
 * fwrite from memory that had already been overwritten, which is a segfault inside
 * ingot with a backtrace that looks like ingot's fault and is not. */
typedef struct { void **buf; int n, cap; } q4x_keep_t;

static int q4x_add(ingot_gguf_writer *gw, const q4x_item_t *it,
                   q4_0_block_t **scratch, size_t *scap, q4x_keep_t *keep) {
    if (!it->w || it->cols % Q4_0_BLOCK_SIZE) {
        fprintf(stderr, "  skip %-34s %s\n", it->name,
                !it->w ? "(weight absent)" : "(K not a multiple of 32)");
        return 0;
    }
    size_t nblk = (size_t)it->rows * (it->cols / Q4_0_BLOCK_SIZE);
    if (nblk * sizeof(q4_0_block_t) > *scap) {
        free(*scratch); *scap = nblk * sizeof(q4_0_block_t);
        *scratch = (q4_0_block_t *)malloc(*scap);
    }
    uint8_t *gbuf = (uint8_t *)malloc(nblk * 18);
    if (!*scratch || !gbuf) { free(gbuf); return 0; }
    if (keep->n == keep->cap) {
        int cap = keep->cap ? keep->cap * 2 : 512;
        void **nb = (void **)realloc(keep->buf, (size_t)cap * sizeof(void *));
        if (!nb) { free(gbuf); return 0; }
        keep->buf = nb; keep->cap = cap;
    }
    keep->buf[keep->n++] = gbuf;

    qwen_quantize_bf16_to_q4_0(it->w, it->rows, it->cols, *scratch);   /* THE function */
    q4_to_ggml(gbuf, *scratch, nblk);

    /* ggml order: ne[0] is the fastest-moving dimension = K */
    uint64_t ne[2] = { (uint64_t)it->cols, (uint64_t)it->rows };
    if (ingot_gguf_add_tensor(gw, it->name, INGOT_TYPE_Q4_0, 2, ne, gbuf) != 0) {
        fprintf(stderr, "  add_tensor failed for %s\n", it->name);
        return 0;
    }
    return 1;
}

int qwen_q4_export_lsq(qwen_tts_ctx_t *ctx, const char *out_path) {
    if (!ctx || !out_path) return 1;
    const int nl = ctx->config.num_layers, h = ctx->config.hidden_size;
    const int inter = ctx->config.intermediate_size;
    const int qd = ctx->config.num_heads * ctx->config.head_dim;
    const int kvd = ctx->config.num_kv_heads * ctx->config.head_dim;
    const int cnl = ctx->config.cp_num_layers, ch = ctx->config.cp_hidden_size;
    const int cinter = ctx->config.cp_intermediate_size;
    const int cqd = ctx->config.cp_num_heads * ctx->config.cp_head_dim;
    const int ckvd = ctx->config.cp_num_kv_heads * ctx->config.cp_head_dim;

    ingot_gguf_writer *gw = ingot_gguf_writer_new();
    if (!gw) return 1;
    ingot_gguf_kv_string(gw, "general.architecture", "qwen3tts-lsq");
    ingot_gguf_kv_string(gw, "general.quantization_algorithm", "qwen weighted-LSQ q4_0 (not llama.cpp RTN)");
    ingot_gguf_kv_u32(gw, "qwen3tts.block_count", (uint32_t)nl);
    ingot_gguf_kv_u32(gw, "qwen3tts.cp.block_count", (uint32_t)cnl);

    q4_0_block_t *sc = NULL; size_t scap = 0;
    q4x_keep_t keep = { NULL, 0, 0 };
    int n = 0;
    char nm[128];

    for (int i = 0; i < nl; i++) {
        const qwen_talker_layer_t *l = &ctx->layers[i];
        const struct { const char *k; const uint16_t *w; int r, c; } S[] = {
            { "attn_q",      l->wq_bf16,   qd,    h     },
            { "attn_k",      l->wk_bf16,   kvd,   h     },
            { "attn_v",      l->wv_bf16,   kvd,   h     },
            { "attn_output", l->wo_bf16,   h,     qd    },
            { "ffn_gate",    l->gate_bf16, inter, h     },
            { "ffn_up",      l->up_bf16,   inter, h     },
            { "ffn_down",    l->down_bf16, h,     inter },
        };
        for (size_t s = 0; s < sizeof S / sizeof S[0]; s++) {
            snprintf(nm, sizeof nm, "blk.%d.%s.weight", i, S[s].k);
            q4x_item_t it = { nm, S[s].w, S[s].r, S[s].c };
            n += q4x_add(gw, &it, &sc, &scap, &keep);
        }
    }
    for (int i = 0; i < cnl; i++) {
        const qwen_cp_layer_t *l = &ctx->cp_layers[i];
        const struct { const char *k; const uint16_t *w; int r, c; } S[] = {
            { "attn_q",      l->wq_bf16,   cqd,    ch     },
            { "attn_k",      l->wk_bf16,   ckvd,   ch     },
            { "attn_v",      l->wv_bf16,   ckvd,   ch     },
            { "attn_output", l->wo_bf16,   ch,     cqd    },
            { "ffn_gate",    l->gate_bf16, cinter, ch     },
            { "ffn_up",      l->up_bf16,   cinter, ch     },
            { "ffn_down",    l->down_bf16, ch,     cinter },
        };
        for (size_t s = 0; s < sizeof S / sizeof S[0]; s++) {
            snprintf(nm, sizeof nm, "cp.blk.%d.%s.weight", i, S[s].k);
            q4x_item_t it = { nm, S[s].w, S[s].r, S[s].c };
            n += q4x_add(gw, &it, &sc, &scap, &keep);
        }
    }
    for (int i = 0; i < 15; i++) {
        snprintf(nm, sizeof nm, "cp.lm_head.%d.weight", i);
        q4x_item_t it = { nm, ctx->cp_lm_head_bf16[i], ctx->config.codebook_size, ch };
        n += q4x_add(gw, &it, &sc, &scap, &keep);
    }
    { q4x_item_t it = { "cp.mtp_proj.weight", ctx->cp_mtp_proj_bf16, ch, ctx->cp_emb_dim };
      n += q4x_add(gw, &it, &sc, &scap, &keep); }
    { q4x_item_t it = { "codec_head.weight", ctx->codec_head_bf16,
                        ctx->config.codec_vocab_size, h };
      n += q4x_add(gw, &it, &sc, &scap, &keep); }
    { q4x_item_t it = { "text_proj.fc1.weight", ctx->text_proj_fc1_bf16,
                        ctx->config.text_hidden_size, ctx->config.text_hidden_size };
      n += q4x_add(gw, &it, &sc, &scap, &keep); }
    { q4x_item_t it = { "text_proj.fc2.weight", ctx->text_proj_fc2_bf16,
                        h, ctx->config.text_hidden_size };
      n += q4x_add(gw, &it, &sc, &scap, &keep); }

    char err[256] = "";
    int rc = ingot_gguf_writer_save(gw, out_path, err, sizeof err);
    ingot_gguf_writer_free(gw);
    for (int i = 0; i < keep.n; i++) free(keep.buf[i]);
    free(keep.buf);
    if (rc != 0) { fprintf(stderr, "write failed: %s\n", err); free(sc); return 1; }
    fprintf(stderr, "Wrote %d tensors (weighted-LSQ Q4_0) to %s\n", n, out_path);

    /* ── the proof the owner asked for, before any audio ──────────────────────────
     * Re-read what was written, undo the ggml nibble order, and compare against a
     * FRESH in-process quantization of the same bf16. Bit-identical or it is not the
     * same quantizer, whatever the file says in its metadata. */
    ingot_gguf *g = NULL;
    if (ingot_gguf_open(&g, out_path, err, sizeof err) != 0) {
        fprintf(stderr, "verify: cannot reopen: %s\n", err); free(sc); return 1;
    }
    int checked = 0, mismatched = 0;
    const struct { const char *name; const uint16_t *w; int r, c; } V[] = {
        { "blk.0.attn_q.weight",    ctx->layers[0].wq_bf16,     qd,    h     },
        { "blk.0.ffn_down.weight",  ctx->layers[0].down_bf16,   h,     inter },
        { "blk.27.ffn_gate.weight", ctx->layers[nl-1].gate_bf16, inter, h    },
        { "cp.blk.0.attn_q.weight", ctx->cp_layers[0].wq_bf16,  cqd,   ch    },
        { "cp.lm_head.0.weight",    ctx->cp_lm_head_bf16[0],    ctx->config.codebook_size, ch },
        { "codec_head.weight",      ctx->codec_head_bf16,       ctx->config.codec_vocab_size, h },
    };
    extern void q4_from_ggml_pub(const uint8_t *src, q4_0_block_t *dst, size_t nblocks);
    for (size_t i = 0; i < sizeof V / sizeof V[0]; i++) {
        const ingot_tensor *t = ingot_gguf_find(g, V[i].name);
        if (!t || !V[i].w) continue;
        size_t nblk = (size_t)V[i].r * (V[i].c / Q4_0_BLOCK_SIZE);
        q4_0_block_t *ref = (q4_0_block_t *)malloc(nblk * sizeof(q4_0_block_t));
        q4_0_block_t *got = (q4_0_block_t *)malloc(nblk * sizeof(q4_0_block_t));
        const uint8_t *raw = (const uint8_t *)ingot_gguf_data(g, t);
        if (ref && got && raw) {
            qwen_quantize_bf16_to_q4_0(V[i].w, V[i].r, V[i].c, ref);
            q4_from_ggml_pub(raw, got, nblk);
            int same = (memcmp(ref, got, nblk * sizeof(q4_0_block_t)) == 0);
            fprintf(stderr, "  verify %-26s %s\n", V[i].name,
                    same ? "BIT-IDENTICAL to --int4's blocks" : "MISMATCH");
            checked++; if (!same) mismatched++;
        }
        free(ref); free(got);
    }
    ingot_gguf_close(g);
    fprintf(stderr, "  %d tensors verified, %d mismatched\n", checked, mismatched);
    free(sc);
    return mismatched ? 1 : 0;
}
