# C12-WIN-10 — Admission slicing: prefill off the established-stream critical path

**Task.** Implementation spec (no code written here). **Question.** How does a newly admitted
request's Talker prefill stop stalling the established streams of its worker, without a
helper thread, without delaying its own first audio more than one slice, and without
changing text semantics? **Known facts.** Inline prefill runs inside the frame loop's
admission block and stalls every active slot for the whole prefill (measured 60-240 ms per
admission on the old engine; the helper arm moved the closed-loop short class 0.966 → 0.915
and pooled 0.923 → 0.915, so the mechanism is real and worth ~0.05); the LOW-priority
helper raised TTFA p95 172 → 683 ms and is NOT the target. **Unknowns.** The per-slice cost
on 8 threads (estimated 0.4-0.5 ms/token from the ~58 ms bf16 prefill of ~120-token texts).

## 1. OBJECTIVE
Bound the stall an admission imposes on established streams to ONE prefill slice (target
<= 25 ms per frame iteration) by running the Talker prefill as resumable token-range slices
inside the frame loop, at most one slice per iteration while other slots are active, all
slices at once when the worker is idle. Nothing else changes: decode quantum, lane, cap,
fail-fast admission, `ADMIT_INSTALL`, prefix cache, text semantics, sampling. Not optimized:
prefill speed itself, TTFA of long texts (it may grow by (slices-1) iterations — measured,
not hidden).

## 2. WHY THIS SHOULD WORK
FACT: `ADMIT_PREFILL` (qwen_tts.c:3154-3176) calls `qwen_tts_generate(ctx, text, NULL, NULL)`
with `ctx->prefill_only=1` synchronously at the top of the iteration, before head/sample/CP/
decode/Talker of every other slot (qwen_tts.c:3339-3425). FACT: the helper arm proved the
short-class tail is this stall (delta −0.051) and that removing it must not cost first audio
(TTFA +511 ms killed it). INFERENCE: a slice of S tokens costs ≈ S × 0.45 ms on 8 threads
(bf16 matmat, `QWEN_PREFILL_CHUNK` 16-token GEMMs); S = 48 → ~20-25 ms, i.e. an admission
becomes one iteration ~+20 ms instead of one iteration +60..240 ms; at ~1 admission/s/worker
the mean cost per iteration is unchanged (~+5 ms) but the per-iteration MAX drops 3-10×,
which is what a p95 over 2.5 s clips sees. HYPOTHESIS: short p95 at C12 drops toward the
helper arm's 0.915 with TTFA p95 staying <= 300 ms.

## 3. EXACT CURRENT DATAFLOW
1. `sink_next_job` pops a request (qwen_tts.c:3339 loop, up to B per iteration).
2. `ADMIT_PREFILL(b, req, prc, pl)` (3154): sets speaker/language, `prefill_only=1`, calls
   `qwen_tts_generate` → builds the prompt (prefix + text tokens → `input_embeds`
   [seq_len][h] f32) → `qwen_talker_prefill(ctx, input_embeds, seq_len)`
   (qwen_tts_talker.c:1408-1805): LAYER-OUTER — for each of 28 layers: q/k/v matmat over all
   tokens in 16-token chunks (`prefill_chunk_tokens`), RoPE at `s + pos0` (pos0 =
   `ctx->pfx_len`, prefix cache), prefix K/V copied per layer from the prefix fill (line
   ~1544), full-sequence f32 attention `qwen_causal_attention_prefill`, K/V converted to bf16
   into `ctx->kv_cache_{k,v}` at `layer*kv_max*kv_dim` (positions [0, seq_len)), o-proj, FFN,
   residual; at the end `ctx->kv_len = seq_len`, `ctx->dec_x` = last hidden.
3. `ADMIT_INSTALL(b, ...)` (3178): memcpy of `pl × kvd` bf16 K and V per layer from
   `ctx->kv_cache_*` into the batch buffers `bb->kv_k/kv_v` of slot b, RMS-norm of `dec_x`
   into `last_hidden[b]`, `pos[b] = pl`; the slot then steps in the same iteration.
Thread ownership: everything on the frame-loop thread, pool width 8 (or 4 while a decoder
unit is in flight). Copies: `ADMIT_INSTALL` is itself ~28 × 2 × pl × 2 KiB (a 300-token
prompt ≈ 34 MB) on the loop thread — keep as is in this task (measured by `admit_ms` minus
`prefill_ms`).

## 4. INTENDED NEW DATAFLOW
Token-range slicing of the SAME computation, TOKEN-OUTER across slices, layer-outer inside a
slice, with the K/V of earlier tokens read from the bf16 cache (exactly what the decode step
already does with `qwen_causal_attention_bf16kv`).

```
pending admission P (per worker, at most cap entries; lives across iterations):
  req, tag, embeds[seq_len][h] (built once by the existing prompt builder), seq_len,
  pos0 (= pfx_len after the prefix KV has been placed in ctx->kv_cache once), done = 0,
  x_slice[S][h] (residual stream of the slice, reused), t_admit, t_first_slice

every iteration, admission block (same place as today):
  if (P exists):
      S = (n_active == 0) ? seq_len - done : min(SLICE, seq_len - done)   # idle worker: no one to protect
      qwen_talker_prefill_range(ctx, P.embeds, P.seq_len, P.done, P.done + S, P.pos0)
      P.done += S
      if (P.done == seq_len): ctx->kv_len = pos0 + seq_len; dec_x = last row → ADMIT_INSTALL as today; P = none
      (exactly one slice per iteration while n_active > 0; no second admission starts until P completes)
  else: pop the next job as today, build embeds, place prefix KV, create P (no compute yet)

qwen_talker_prefill_range(ctx, embeds, seq_len, t0, t1, pos0):
  n = t1 - t0; x = embeds[t0:t1] (copy into x_slice)
  for layer in 0..27:
      x_norm = rms(x)                                   # n rows
      q,k,v = matmat(x_norm)                            # existing 16-token chunk GEMMs, rows n
      rope(q,k, position = pos0 + t0 + s)               # s in [0,n)
      write k,v (bf16) into ctx->kv_cache_{k,v}[layer][pos0+t0+s]   # same layout as today
      for s in 0..n-1:                                  # attention over the cache, causal
          attn[s] = qwen_causal_attention_bf16kv(q[s], cache_k[layer], cache_v[layer], len = pos0+t0+s+1)
      x = x + oproj(attn); x = x + ffn(rms(x))          # existing code, rows n
  if t1 == seq_len: ctx->dec_x = final_norm_input of x[n-1] as the current function leaves it
```
Shapes: q/k/v `[n][q_dim|kv_dim]` f32; cache rows bf16 as today. Ownership: frame-loop thread,
pool width as the iteration finds it. No new buffers except `x_slice` and the per-slice q/k/v
(<= S rows). Prefix: at P creation copy the prefix fill's bf16 K/V into `ctx->kv_cache` rows
[0,pos0) once (the current code copies f32 prefix K/V per layer at line ~1544; here the
prefix lives in the cache in bf16, which is the precision the decode path already attends
over). Eligibility for the first frame: the iteration in which the last slice completes (as
today). Starvation: a pending P always gets exactly one slice per iteration — established
streams cannot starve it, and it cannot take more than one slice from them. Bounded state:
one P per worker; the parent's fail-fast cap is untouched, the reader queue is unchanged.
Text semantics: the prompt and token order are byte-identical to today; nothing is split.
What this does NOT deliver: TTFA independent of text length (that is the incremental-text
track, a different item); here a long text pays (slices−1) extra iterations of TTFA, which
the gate below bounds.

## 5. FORBIDDEN IMPLEMENTATIONS
* NOT the helper thread / LOW priority (`QWEN_PREFILL_HELPER`, `QWEN_PREFILL_LOW_MS`): it
  trades TTFA for the stall; keep it as the falsifier only.
* NOT running the prefill on the decoder lane or on a private team.
* NOT splitting the text into sentences or trimming the prompt to get a smaller first slice.
* NOT more than one slice per iteration while `n_active > 0`; NOT slicing when idle.
* NOT deferring `ADMIT_INSTALL` or changing `pos[b]`, `tcl[b]`, `last_hidden` semantics.
* NOT keeping f32 K/V of all layers for the whole request across slices (69 MB per 300
  tokens); the earlier tokens' K/V are read from the bf16 cache.
* NOT a new queue between the reader and the loop; NOT touching `--max-queue`, cap, M1.
* NOT recomputing earlier tokens per slice (O(n²)); each token is prefilled once.

## 6. IMPLEMENTATION SKETCH
```
/* qwen_tts_talker.c */
int qwen_talker_prefill_range(qwen_tts_ctx_t *ctx, const float *embeds, int seq_len,
                              int t0, int t1, int pos0);
/* Body = the per-layer body of qwen_talker_prefill restricted to rows [t0,t1):
   reuse prefill_chunk_tokens() GEMMs (rows n), apply_rope_neox_inplace(.., pos0+t0+s),
   f32_to_bf16_vec into ctx->kv_cache_k/v + layer*kv_max*kv_dim + (pos0+t0)*kv_dim,
   then per row s: qwen_causal_attention_bf16kv(out_row, q_row, cache_k_layer, cache_v_layer,
   pos0+t0+s+1, ...)  -- the decode-step attention, check its exact signature in
   qwen_tts_talker.c before use; if it only takes one query, call it n times. */

/* qwen_tts.c: new per-worker pending-admission struct next to the admission block */
typedef struct { qwen_batch_req_t req; void *tag; float *embeds; int seq_len, pos0, done;
                 double t_admit; int slot; } adm_pending_t;
static int  adm_slice_tokens(void);      /* QWEN_PREFILL_SLICE, default 0 = today's inline path */
/* admission block: if (adm.active) run one slice (or all when n_active==0) else pop+prepare.
   Preparing = the existing prompt/embedding builder split out of qwen_tts_generate's
   prefill_only path (function that returns embeds+seq_len+pos0 without running the Talker). */
```
Flag `QWEN_PREFILL_SLICE=N` (0 = current code path, byte-identical); register it in
`g_qwen_reported_flags` (qwen_tts_kernels.c:322-352) and `qwen_flag_scope.h`; add a
dispatch row `talker.prefill.slice` in qwen_tts_dispatch.c. `[STAGE]` already has
`admit_ms`/`prefill_ms`; add `prefill_slice=<S>` to the line. Cleanup: free `embeds` on
completion or on cancel/disconnect of the pending request (the pending P must honour
`sink_cancelled` like a slot: if cancelled, drop P and free).

## 7. FILES / FUNCTIONS TO TOUCH
* `qwen_tts_talker.c`: add `qwen_talker_prefill_range` (new function; do not edit
  `qwen_talker_prefill` — it remains the control path).
* `qwen_tts.c`: admission block (3339-3425) and `ADMIT_PREFILL` split into prepare/run;
  `adm_pending_t`; STAGE field. `ADMIT_INSTALL` unchanged.
* `qwen_tts_kernels.c` (flag registry), `qwen_flag_scope.h`, `qwen_tts_dispatch.c` (row).
* Tests: `tests/prefill_slice_parity.py` (new, see §9).
Do NOT touch: decoder lane, `dec_enqueue`, quantum/ramp, server parent, profiles,
`qwen_tts_server.c` reader/dispatch, sampling, M1 early admission, helper path.

## 8. NUMERICAL / SEMANTIC INVARIANTS
* Prompt tokens, positions and RoPE angles identical to the inline path.
* K/V written to the cache at the same rows with the same bf16 conversion.
* Attention of a new token over earlier tokens uses bf16 K/V instead of f32 — NOT EXACT;
  bounded: the decode path already attends over the same bf16 cache. Gate in §9.
* `dec_x` after the last slice equals the inline `dec_x` within bf16-attention error.
* No change to what the client receives: same request semantics, same fail-fast boundary,
  same quantum/ramp. No hidden queue: at most one pending P per worker, visible in
  `[serve-profile]` as `admissions_pending_max`.
* `QWEN_PREFILL_SLICE` unset → byte-identical binary behaviour to today.

## 9. LOCAL CORRECTNESS ORACLE (before any cloud run)
1. Build + `--self-test` + `check_flag_registry` + `perf_profile.py validate` PASS.
2. `tests/prefill_slice_parity.py`: single worker (`--prefork 1 -j 8`), `--batch-size 1`,
   temperature 0, seed 42, 10 bank texts (short/medium/long/italian), arm A inline, arm B
   `QWEN_PREFILL_SLICE=48`: (a) the first 8 codec frames' codes identical for >= 9/10 texts;
   (b) `tests/compare_audio.py` per pair mel-corr >= 0.99, duration within 2 %; (c) zero
   errors, `wav_qc` equal. Also `QWEN_PREFILL_SLICE=1000` (one slice) must be exactly the
   inline result modulo bf16 attention (same gate).
3. Cancel/disconnect during a pending P (`tests/cancel_correctness.py` with a long text and
   slice 16): no leak (ASan build), no orphaned P, worker stays healthy.

## 10. MICROBENCH DESIGN
No kernel microbench (this is a scheduling change). Server DIAGNOSTIC on the Turin host,
one worker `1x8@0-7`, cap 3, `QWEN_STAGE_TRACE=1`, closed-loop 3 clients on the mixed bank
for 3 minutes, arms inline vs `QWEN_PREFILL_SLICE=48`: metric = distribution of
`admit_ms` per iteration. GO to the server A/B only if `admit_ms` p95 <= 30 ms in the slice
arm (control 60-240) AND the slice arm's per-request prefill total is within +15 % of inline.

## 11. SERVER A/B GATE
Frozen `turin-c8a-32c-vnni-product` (only `QWEN_PREFILL_SLICE=48` added on the treatment
via `--server-env`; the profile must list it under `tunable_flags` for the preflight, or the
run is a two-profile A/B as for RES1_V2), 4x8 cap 4, C12, 10-minute closed-loop soak each,
same bank, temperature 0.9. PASS if: short p95 <= 0.92 (control 0.959-0.966), conversational
<= 0.90, pooled <= 0.90, TTFA p95 <= 300 ms, safe-start p95 <= 500 ms, prebuffer p95 <= 300,
stall@250 <= 0.5 %, stall@500 = 0, errors/timeouts 0. Then the C12-WIN-8 qualification.

## 12. STOP / REVERT RULES
STOP after the diagnostic if `admit_ms` p95 does not drop below 30 ms, or the slice arm
does more total prefill work than inline (+15 %), or codes parity (<9/10) fails.
REVERT immediately if TTFA p95 > 350 ms, safe-start p95 > 550, stall@250 > 0.5 %, any
error/timeout, or the parity gate fails. One allowed correction: SLICE 48 → 32 or 64, once.
Never make the helper, a thread, or a smaller prompt the "fix".

## 13. SUCCESS STATE
Commit "serve: slice the Talker prefill across frame iterations (QWEN_PREFILL_SLICE)" with
the new function, flag, dispatch row, parity test; evidence in
`.work/c12-win-admission-slicing-<date>.md` (diagnostic table, A/B table, parity output);
PLAN C12-WIN-10 closed with numbers; profile gains `QWEN_PREFILL_SLICE=48` ONLY after the
C12-WIN-8 qualification (status provisional → qualified requires the ear check as usual).
