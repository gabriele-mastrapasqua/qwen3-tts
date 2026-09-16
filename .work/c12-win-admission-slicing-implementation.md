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

## 8. CORRECTNESS CONTRACT (REVISED 2026-09-10 — supersedes the original 8 and 9.2)

The original sections 8 and 9.2 were internally inconsistent and the local implementation
proved it. Section 5 forbids keeping the f32 K/V of all layers across slices; section 9.2
then required the sliced result to match the monolithic one to mel-corr >= 0.99 with
identical codes. Those cannot both hold: slices are token-outer and layer-inner, so at
slice 2 layer 5 needs layer 5's K/V of the slice-1 tokens, which slice 1 computed and
discarded. Reading them back from the bf16 cache is the only option section 5 leaves, and
it is not bit-equal to an f32-resident pass. **Section 9.2 is SUPERSEDED. It is not
relaxed — it is withdrawn as unsatisfiable, and replaced by C below.**

Three different notions of correctness apply, and they must never be collapsed.

### A. SLICING STATE-MACHINE PARITY — HARD EXACT GATE

For any two valid partitionings of the SAME sliced computation, uninterrupted versus
paused-and-resumed, the following must be EXACT, not approximate:

* `dec_x` (the final hidden the first decode step consumes) — bit-equal
* the K cache over `[0, kv_len)`, every layer — bit-equal
* the V cache over `[0, kv_len)`, every layer — bit-equal
* `kv_len` — equal
* the Talker pre-generation state as a whole — equal

This is the primary structural oracle. **Any failure here is an implementation bug**, never
a tolerance to widen. Oracle: `--prefill-slice-check <S>`, which reports each arm against
the unsliced arm and against the first sliced arm; the second comparison is the one that
isolates slicing from precision, because both arms attend over the same bf16 cache.

Measured 2026-09-10 (0.6B, 38- and 107-position prompts), slices 2, 3, 4, 5, 7, 16, 32, 48:
`dec_x` 0.000e+00, zero K rows differing, zero V rows differing, `kv_len` equal. PASS.

Boundary condition that is part of the contract: **no emitted slice may contain exactly one
token.** At M=1 the shared prefill projection kernels take the matvec path, whose
accumulation order differs from the matmat path, which made the result depend on where the
boundaries fell. `qwen_prefill_slice_next` absorbs a 1-token remainder into the current
slice (so a slice is at most `S+1` tokens and never one) and is shared by the serving loop
and the oracle so the gate exercises the shipped rule. `QWEN_PREFILL_SLICE=1` is a debug
value only; it crosses that kernel boundary by construction and is not held to A.

### B. MONOLITHIC-vs-SLICED NUMERICAL DRIFT — EXPECTED, BOUNDED, NOT A BUG

The sliced continuation reads earlier tokens from the bf16 KV cache; the monolithic pass
keeps f32 K/V resident for the whole prompt. The two therefore differ numerically. This is
a property of the design, not a defect, and it must not be reported as one.

Measured drift of the final pre-generation state:

| prompt | dec_x max abs / max ref | K rows differing | V rows differing |
|---|---|---|---|
| 38 positions | 7.96e-04 | 783/1064 | 783/1064 |
| 107 positions | 1.686e-03 | 2646/2996 | 2646/2996 |

Both are below one bf16 ulp (3.91e-3), i.e. at the quantisation floor of the cache the
sliced path attends over. **Do not treat this as a state divergence bug.** But do not
dismiss it either: at temperature 0 a difference far below an ulp is enough to flip one
argmax, after which the utterance diverges completely. Small in state space is not small in
output space once a discrete choice sits downstream.

Worth recording because it reframes which arm is anomalous: during a monolithic prefill a
prompt token attends over f32 K/V, but every decode step afterwards attends over the bf16
cache. The sliced path is the self-consistent one; the control is the special case.

### C. PRODUCT QUALITY PARITY — THE QUALIFICATION GATE

Because A cannot be extended across the monolithic/sliced boundary, acceptance is a
QUALITY regression gate on real generated audio, not a state oracle. Paired runs, same
text, seed, speaker/voice path, model, precision and server configuration; treatment differs
only by `QWEN_PREFILL_SLICE`.

Per pair, in this order of authority:

1. **Valid WAV** — parses, non-empty, sane sample rate, no clipping run, no silence-only
   tail. Automated `wav_qc`. Any failure is a hard stop.
2. **Duration semantics** — the sliced arm produces a different but valid utterance, so
   duration is NOT expected to be identical. Gate the DISTRIBUTION: median |Δdur| and the
   worst case, against the control arm's own run-to-run spread on the same bank. A pair
   outside the control's spread is an outlier to listen to, not an automatic failure.
3. **ASR transcript comparison** — CER/WER per pair against the input text, control and
   treatment scored the SAME way. The gate is that the treatment's CER distribution is not
   worse than the control's (report median, p90, max, and every pair where treatment CER
   exceeds control CER by more than the control's own p90 spread). ASR is never the sole
   oracle: it has its own error floor and it is language-dependent.
4. **mel / log-mel similarity** — SOFT DIAGNOSTIC ONLY. It is not a proof of anything here,
   because two valid utterances of the same sentence legitimately score low. Use it to RANK
   pairs for listening, never as a pass/fail.
5. **Human listening** on every pair flagged by 2, 3 or 4, and on a fixed random subset.

Bank: small and paired, 20-30 sentences, reported as a distribution plus a named outlier
list. Do not compute a single mean and call it a gate. Do not invent a permissive
threshold: where no defensible threshold exists, report the distribution and escalate.

The product question C exists to answer is exactly:

> Does sliced admission preserve acceptable speech quality while materially reducing
> established-stream interference, WITHOUT repeating the PREFILL_HELPER TTFA catastrophe
> (172 -> 683 ms)?

Quality alone is not a pass, and interference reduction alone is not a pass. Both, or no.

## 9. LOCAL CORRECTNESS ORACLE (before any cloud run)

1. Build + `--self-test` + `make check-flag-registry` + `tools/flag_parity.py --check` +
   `tests/test_perf_profile.py` + `tools/check_plan.py` PASS.
2. `--prefill-slice-check <S>` for S in {2, 3, 16, 48}: contract A, exact. This is the gate
   that decides whether the implementation is correct.
3. `tests/prefill_slice_parity.py`: asserts FIRST that the treatment actually took the
   sliced path (`[ADMSLICE] first_sliced_admission`) and that the control did not. Without
   that check the whole harness is vacuous — a WAV is a function of integer codes, so a
   correct slicing and a slicing that never ran produce identical files. It also requires
   `--batch-size >= 2`, because 1 routes to the non-batched server where the admission
   block does not exist.
4. Cancellation: drop a pending admission and serve again. Currently INCONCLUSIVE — the
   post-cancel request is refused with 503 on the CONTROL arm too (PLAN TQ-2, fail-fast
   admission). It must be re-run once TQ-2 is fixed; it is not a verdict on this change.

## 10. MEASUREMENT DESIGN (REVISED 2026-09-10)

No kernel microbench: this is a scheduling change. Spec 10 is measured SEPARATELY from the
Spec 11/12 winners — one mechanism at a time — and only after those have their own verdict,
so a decoder change never sits inside a scheduler A/B.

CONTROL: current monolithic admission (`QWEN_PREFILL_SLICE` unset).
TREATMENT: `QWEN_PREFILL_SLICE=48`. Everything else identical: same profile, topology, cap,
quantum, lane, bank, seed policy, model path, precision.

Workloads, in this order:

1. **Isolated new request** — short / medium / long input, no other traffic. Establishes
   TTFA cost of slicing on its own; a long text pays (slices-1) extra iterations and that
   must be visible, not pooled away.
2. **One established stream + a new SHORT request.**
3. **One established stream + a new LONG request** — the worst case for interference and
   the one the mechanism exists for.
4. **Steady closed-loop short workload** — the class the helper arm moved 0.966 -> 0.915.
5. **Mixed workload.**

Metrics, per arm and per workload:

* new-request TTFA (p50/p95/max)
* established-stream MAX GAP — the direct measure of the interference this removes
* `required_prebuffer`, `safe_play_start`
* STREAM_RTF p50/p95, TOTAL_RTF where useful
* stall@250, stall@500
* errors / rejects / timeouts
* `admit_ms` distribution per iteration (the mechanism's own metric)
* slice count per request and slice wall time (`[ADMSLICE]`, `[STAGE] prefill_slice=`)
* **monolithic fallback count** — reported separately, never pooled

**The cold prefix-cache population request MUST be reported separately from steady sliced
admissions.** The first request that populates a prefix slot falls back to the monolithic
path by design (contract in section 8 / the implementation record). Pooling it into the
same distribution would both understate the slicing benefit and hide a regression in the
fallback. Split every table into `steady sliced` and `cold fallback`.

GO to the server A/B only if, in the treatment: `admit_ms` p95 <= 30 ms (control 60-240),
total per-request prefill work within +15 % of the control, and no increase in rejects.

## 11. SERVER A/B AND QUALIFICATION GATE
Frozen `turin-c8a-32c-vnni-product` (the treatment adds only `QWEN_PREFILL_SLICE=48`; the
profile must list it under `tunable_flags` for the preflight, or run it as a two-profile
A/B as was done for RES1_V2), 4x8 cap 4, C12, 10-minute closed-loop soak per arm.

Performance PASS: short p95 <= 0.92 (control 0.959-0.966), conversational <= 0.90, pooled
<= 0.90, TTFA p95 <= 300 ms, safe-start p95 <= 500 ms, prebuffer p95 <= 300, stall@250
<= 0.5 %, stall@500 = 0, errors/timeouts 0.

Quality PASS: contract C of section 8 — the paired quality bank, reported as a distribution
with named outliers. Performance without quality is not a pass, and quality without the
interference reduction is not a pass either.

## 12. STOP / REVERT RULES
STOP after the diagnostic if `admit_ms` p95 does not drop below 30 ms, or the treatment does
more total prefill work than the control (+15 %).
STOP and report, do not adjust, if contract A (section 8) fails at any slice size >= 2: that
is an implementation bug, not a tuning question.
REVERT immediately if TTFA p95 > 350 ms, safe-start p95 > 550, stall@250 > 0.5 %, or any
error/timeout appears. One allowed correction: SLICE 48 -> 32 or 64, once.
Never make the helper, a thread, or a smaller prompt the "fix".
Do NOT use contract B (monolithic-vs-sliced drift) as a stop reason; it is expected.

## 13. SUCCESS STATE
Implementation: DONE locally (2026-09-10), default off, contract A exact. Remaining:
the section 10 diagnostic, the section 11 A/B, and the contract C quality bank on the
qualification model path. `QWEN_PREFILL_SLICE` enters the product profile only after all
three, and status stays `provisional` until the ear check as usual.
