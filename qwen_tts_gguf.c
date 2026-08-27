/* qwen_tts_gguf.c — load Talker weights from a quantized GGUF file.
 *
 * WHY THIS EXISTS, AND WHY IT IS THIS SMALL
 * -------------------------------------------------------------------------
 * The engine loads safetensors and quantizes at load time. A GGUF carries
 * weights ALREADY quantized by llama.cpp's algorithms (Q8_0, Q6_K, Q4_K_M,
 * Q4_0), which are not ours: the K-quants run weighted least squares plus a
 * grid search, we run absmax (plus weighted LSQ for q4_0). The question this
 * file answers is: *how does our engine sound when the Talker weights are
 * exactly what GGUF Q6_K holds?* — the quality gate, before we commit to an
 * internal format.
 *
 * This is not the full GGUF loader. It is the smallest patch that carries ONE
 * format end to end, by reusing two things the engine already has:
 *
 *   1. `ingot` (third_party/, already linked into every target) reads GGUF
 *      v2/v3 and decodes 33 of the 34 ggml block types. No parser to write.
 *   2. The override registry (`qwen_track_override`) lets us replace a pointer
 *      into the mmapped weights with heap memory that `qwen_tts_unload` frees.
 *
 * WHAT IT REPLACES, AND WHAT IT DOES NOT
 * Only the seven per-layer matrices: q/k/v/o and gate/up/down. Those are what
 * quantization acts on and where the formats differ. Left untouched from the
 * original checkpoint: embeddings, norms, codec_head, and the whole Code
 * Predictor — which in the GGUF are either F32 anyway or structurally
 * transformed (llama.cpp folds `text_projection` into the embedding table and
 * concatenates `codec_embedding`, and undoing that folding would not help
 * answer the question).
 *
 * THE NAME MAP is 1:1 and needs no heuristics:
 *     blk.<N>.attn_q.weight    <->  talker.model.layers.<N>.self_attn.q_proj.weight
 *     blk.<N>.ffn_down.weight  <->  talker.model.layers.<N>.mlp.down_proj.weight
 *
 * TENSOR SHAPE: HF stores linear weights as [out, in]; GGUF keeps the same
 * bytes with ne = [in, out]. Both are row-major, so `ingot_gguf_shape_row_major`
 * hands back [out, in] and matches us with no transpose. The check below
 * verifies that instead of assuming it: a mismatch is an error, not a warning.
 *
 * PRECISION: we dequantize to f32 and convert to bf16, the format the engine
 * keeps dense weights in. bf16 rounding (~0.4%) sits an order of magnitude
 * below the quantization error we are measuring (0.55% for Q8_0, 6.5% for
 * Q4_K_M), so it does not contaminate the comparison — but it is worth stating
 * rather than assuming.
 */
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

/* f32 -> bf16, round-to-nearest-even.
 * Plain truncation would cost half an ULP of systematic bias on every weight.
 * Across 1.6 billion weights that is not noise, it is drift. */
static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(x >> 16); /* NaN/Inf: truncate */
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return (uint16_t)(x >> 16);
}


/* ── The GGUF Q4_0 native path: a nibble permutation, not a new kernel ──────────
 *
 * Our q4_0_block_t and ggml's block_q4_0 are the SAME 18 bytes for 32 weights: an
 * fp16 scale followed by 16 packed nibble pairs, both biased by +8. Only the pairing
 * differs, and that is the whole conversion:
 *
 *     ggml:  qs[j] = q[j] | (q[j+16] << 4)      -- low half with high half
 *     ours:  qs[i] = q[2i] | (q[2i+1] << 4)     -- adjacent pairs
 *
 * The adjacency is not arbitrary: our kernel turns one 16-byte load into 32 int8 in
 * value order with a single `vzipq_u8(vandq_u8(raw,mask), vshrq_n_u8(raw,4))`
 * (qwen_tts_kernels.c:4792). Reading ggml's order would need a `vuzp` on every call,
 * on every block, forever. Permuting once at load costs one pass over the file and
 * buys the existing SMMLA/NEON q4 kernels unchanged.
 *
 * This IS a repack, in its smallest honest form — the same idea llama.cpp applies at
 * set_tensor time, applied here to a layout we already had. */
void q4_from_ggml_pub(const uint8_t *src, q4_0_block_t *dst, size_t nblocks);
static void q4_from_ggml(const uint8_t *src, q4_0_block_t *dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t *s = src + b * 18;
        memcpy(&dst[b].scale_f16, s, 2);          /* fp16 scale, byte-identical */
        const uint8_t *qs = s + 2;
        for (int i = 0; i < 16; i++) {
            /* q[k] with k = 2i and 2i+1, read from ggml's split-half packing */
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

    /* The seven matrices, with the shape the engine expects. */
/* `q4off` is the matching q4 slot; `fuse` says the rows land interleaved in
     * gate_up_fused_q4 (row r of gate -> 2r, row r of up -> 2r+1), which is the
     * layout the engine's own --int4 path builds and every q4 kernel expects.
     * fuse: 0 = its own array, 1 = even rows of the fused one, 2 = odd rows. */
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

    /* One f32 scratch sized on the largest tensor, allocated once rather than
     * per tensor: ~50 MB on a 1.7B, against 196 allocations. */
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
    /* Live-bytes accounting, per representation. The question this answers: how many
     * copies of the same matrix are resident at once, and which of them is ever read.
     *   prefill bf16 : the ORIGINAL safetensors mapping - not allocated here, but
     *                  resident because prefill touches every page.
     *   dequant bf16 : allocated below, and read ONLY if a quantized path declines.
     *   blocks       : our own q4_0_block_t copy, read by our q4 kernels.
     *   packed       : KleidiAI RHS / q8_0x4 repack, read by those kernels. */
    size_t by_pref = 0, by_dequant = 0, by_blocks = 0;
    const char *first_type = NULL;

    for (int li = 0; li < nl; li++) {
        qwen_talker_layer_t *l = &ctx->layers[li];
        /* The fused gate+up prefill copy: snapshot the ORIGINAL before the rebuild below
         * overwrites it with the quantized gate/up. One allocation per layer, freed by
         * the override registry. */
        if (!gguf_quant_prefill() && !l->gate_up_fused_bf16_pref && l->gate_up_fused_bf16) {
            size_t n = (size_t)(2 * inter) * h;
            uint16_t *cp = (uint16_t *)aligned_malloc(n * sizeof(uint16_t));
            if (cp) {
                memcpy(cp, l->gate_up_fused_bf16, n * sizeof(uint16_t));
                l->gate_up_fused_bf16_pref = cp;
                qwen_track_override(ctx, cp);
            }
        }
        uint8_t *fused_ggml = NULL;   /* interleaved gate+up in ggml layout, transient */
        uint8_t *fused_q8 = NULL;     /* same, for the Q8_0 path */
        int fused_bpr = 0;
        for (int s = 0; s < nslots; s++) {
            char name[128];
            snprintf(name, sizeof name, "blk.%d.%s.weight", li, slots[s].kind);

            const ingot_tensor *t = ingot_gguf_find(g, name);
            if (!t) { missing++; continue; }

            uint64_t shape[INGOT_MAX_RANK] = {0};
            ingot_gguf_shape_row_major(t, shape);
            if ((int)shape[0] != slots[s].rows || (int)shape[1] != slots[s].cols) {
                /* Shape differs from what the engine expects: do NOT guess.
                 * Skipping loudly beats writing crooked weights silently. */
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
            by_pref += n * sizeof(uint16_t);   /* the original, resident via mmap */

            /* Q4_0 does not need a dequantized bf16 at all, and used to allocate one:
             * 2.8 GB on this model, written once and never read. Decode goes to the q4
             * blocks (they win the dispatch priority) and prefill reads the ORIGINAL
             * through *_bf16_pref. Leaving l->wq_bf16 pointing at the original also
             * makes the fallback BETTER than it was - full precision instead of
             * 4-bit-rounded values. */
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
            /* QWEN_GGUF_KEEP_BF16=1 leaves the ORIGINAL safetensors bf16 in place and
             * installs only the quantized blocks. It exists to separate two things
             * that --int4 and the GGUF path confound: --int4 quantizes for DECODE but
             * leaves prefill reading full-precision bf16, while the GGUF path replaces
             * the bf16 with the dequantized 4-bit values and so runs prefill at 4 bits
             * too. The prompt - speaker token, language, text - is built in prefill. */
            /* Record the ORIGINAL as the prefill weight before replacing the decode
             * one. This is the fix for the 2026-08-22 collapse, made structural rather
             * than optional: prefill decides the conditioning and stays full precision,
             * decode carries the quantization. QWEN_GGUF_QUANT_PREFILL=1 restores the
             * old (broken) behaviour for A/B only. */
            if (!gguf_quant_prefill()) {
                switch (slots[s].fuse) {
                    case 1: case 2: break;   /* gate/up: the fused prefill copy is set below */
                    default: {
                        const uint16_t **pref = (const uint16_t **)((char *)l + slots[s].pref_off);
                        if (!*pref) *pref = *field;
                        break;
                    }
                }
            }
            if (bf) { *field = bf; qwen_track_override(ctx, bf); }
            applied++;

            /* Native Q4_0: keep the weights AS BLOCKS as well, so the decode step
             * dispatches to the q4 SMMLA/NEON kernels instead of reading bf16. The
             * bf16 copy above stays because prefill's matmat path reads bf16
             * directly (qwen_tts_talker.c:1422) — without it prefill would silently
             * keep using the ORIGINAL safetensors weights and the render would be a
             * mix of two checkpoints. Dropping the bf16 is a later step, once the
             * prefill path can consume blocks too. */
            /* Q8_0: keep the per-block-32 form and run it on the repacked ARM path.
             * Registered under `bf` - the pointer the engine's kernels will be called
             * with - so no dispatcher needs a new branch. gate/up are two tensors the
             * engine consumes as one interleaved matrix, so they are assembled first. */
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
                            /* interleave: source row r -> destination row 2r (+1 for up) */
                            int odd = (slots[s].fuse == 2);
                            for (int r = 0; r < slots[s].rows; r++)
                                q4_from_ggml(raw + (size_t)r * bpr * 18,
                                             *q4f + (size_t)(2 * r + odd) * bpr, (size_t)bpr);
                        }
                        /* Verify the permutation on the FIRST tensor, against the value
                         * ingot already dequantized into `f32`. Both describe the same
                         * weights, so the difference must be exactly zero — not "small".
                         * A nibble permutation that is subtly wrong produces plausible
                         * audio, which is the worst failure mode there is. */
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
                        /* KleidiAI wants the ggml bytes, NOT our re-nibbled ones: its
                         * `qsu4c32s16s0` source layout IS block_q4_0. So it is fed from
                         * `raw`, and keyed by the pointer the kernels will be called
                         * with (`*q4f`), which is how the dispatcher finds it later.
                         *
                         * gate and up are two separate GGUF tensors that the engine
                         * consumes as ONE interleaved matrix, so the packer needs an
                         * interleaved ggml-layout copy. It is built once per layer,
                         * packed, and freed - transient, not resident. */
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
                                if (odd) {   /* `up` is the second of the pair: now complete */
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
        free(fused_ggml);   /* non-NULL only if `up` was missing: never leak the pair */
        free(fused_q8);
    }
    free(f32);

    /* The fused gate+up copy is built at load time from the OLD pointers. Without
     * rebuilding it the engine would keep using the pre-override weights and the
     * GGUF would not change a single note. Same code as the .expr path in main.c. */
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
            /* Correctness before speed: compare KleidiAI against our own q4 kernel on
             * layer 0's Q projection. They quantize the activation differently (per-32
             * block vs per-vector), so bit-equality is not the bar - agreement to a
             * fraction of a percent is. A gross mismatch means the packing is wrong,
             * and a wrong packing still produces confident, plausible audio. */
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

/* ── The Code Predictor, and the two Talker matrices the main GGUF does not carry ──
 *
 * WHY A SECOND FUNCTION AND NOT A GENERALISED FIRST ONE
 * The two artifacts have different shapes of problem. The Talker GGUF comes from
 * llama.cpp's own converter, so its names are llama.cpp's (`blk.N.attn_q.weight`) and
 * its tensor set is fixed by that tool. This one we write ourselves
 * (`tools/make_rest_gguf.py`), so the names are ours and the set is exactly what the
 * engine can consume. Folding both into one loop would mean a name-mapping table with
 * two dialects and a lot of conditionals, to save maybe thirty lines.
 *
 * WHY THE CODE PREDICTOR MATTERS MORE THAN ITS SIZE SUGGESTS
 * It is 175 Mparam against the Talker's 1409 - 8% of the weights - but it runs 15
 * times per frame, and measured on this box it costs 26.5 ms/frame against the
 * Talker's 9.4. Roughly three quarters of decode time lives here.
 */
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

    /* The scratch has to fit the LARGEST tensor this function touches, and that is
     * codec_head [codec_vocab, talker_hidden], not an lm_head. */
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

    /* One tensor: dequantize -> bf16 (the path prefill and any non-q4 kernel reads),
     * keep the Q4_0 blocks natively, and hand the RAW ggml bytes to KleidiAI. */
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
        /* Same saving as the Talker loader: a Q4_0 tensor needs no dequantized bf16.     \
         * The blocks win the dispatch and the original stays as the fallback. */         \
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
                q4_0_block_t **q4dst = (Q4_FIELD); /* NOT q4f: caller passes a var so named */                                                  \
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
        /* Same trap as the Talker: the fused gate+up bf16 copy was built at load from
         * the OLD pointers, so without rebuilding it the CP would keep using the
         * pre-override weights and the GGUF would change nothing audible. */
        if (l->gate_bf16 && l->up_bf16 && l->gate_up_fused_bf16) {
            size_t row_bytes = (size_t)h * sizeof(uint16_t);
            for (int r = 0; r < inter; r++) {
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r)*h,   l->gate_bf16 + (size_t)r*h, row_bytes);
                memcpy(l->gate_up_fused_bf16 + (size_t)(2*r+1)*h, l->up_bf16   + (size_t)r*h, row_bytes);
            }
        }
    }

    /* The 15 lm_heads: [codebook_size, cp_hidden], one per codebook, all hit every
     * frame - which is exactly why the CP costs what it costs. */
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
    /* The three matrices that live outside the per-layer blocks. They are in the
     * artifact and were, until now, quantized-but-unused - the exact residue that
     * separated this from a FULL Q4_0 without asterisks. */
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

/* Public wrapper: the Q4 exporter verifies its own output by undoing this exact
 * permutation, so the two must be the same code, not two copies of it. */
void q4_from_ggml_pub(const uint8_t *src, q4_0_block_t *dst, size_t nblocks) {
    q4_from_ggml(src, dst, nblocks);
}

/* ── MODEL SOURCES: the banner that makes a benchmark self-documenting ────────────
 *
 * Printed after every load. It exists because a firing kernel proves the FORMAT and
 * says nothing about the PROVENANCE: `CP q8 repack 100%` is equally true whether the
 * weights came from the GGUF artifact or from safetensors quantized at load. Over a
 * long session it is easy for one command to carry --gguf-talker and drop --gguf-rest,
 * and for the resulting table to be believed.
 *
 * Anything not claimed by a GGUF loader is reported as what it actually is.
 */
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
    /* KleidiAI does not change WHERE a weight came from, it changes what runs on it -
     * but the packed copy is real memory and a benchmark that reports RSS has to
     * name it. Zero everywhere the ISA or the flags say no. */
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
    /* The line that answers the question directly: anything eligible NOT taken from a
     * GGUF is still on the old path, and it is named rather than implied. */
    int cp_missing = ctx->src.cp_n ? (cp_elig - ctx->src.cp_n) : 0;
    int head_missing = ctx->src.cp_heads_n ? (15 - ctx->src.cp_heads_n) : 0;
    if (ctx->src.cp_n || ctx->src.talker_n)
        fprintf(stderr, "  CP safetensors quantized fallback: %d layer tensors, %d lm_heads\n",
                cp_missing, head_missing);
    fflush(stderr);
}
