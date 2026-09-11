# DL-4 multi-slot — the Arm twin of Design-D, design and acceptance

**Task.** ARM-LINUX-V2 item 3: batch more than one slot's decoder unit into one DL-4 pass
so the weight sweep is paid once for all slots, the way x86 AMX Design-D/ragged does.
**Question.** What exactly must change in the kernel and in the lane so the batch is exact,
and how is the gain proven?
**Known facts.** Sections 1-3, read from the tree at `c2a638a`.
**Unknowns.** The measured gain; the register pressure outcome on Neoverse-V2; whether the
lane can hold two units of the same owner without breaking the lead bound.

## 1. Why it is two changes, not one

The DL-4 kernel (`qwen_conv1d_int8_v2_ctx`, both leaves) already owns everything an exact
batch needs: per-position activation scales (`sc[r]`), per-(channel,tap) weight scales, the
left context via `(tail, tail_cols)`, and the residual epilogue. What it does NOT have is a
slot dimension: `sd_dconv_job_t` carries one `in`, one `tail`, one `out`.

The lane (`qwen_lane_*`, `qwen_tts_thread.c`) runs ONE unit in flight per slot and blocks a
slot only when it needs another quantum. Two units from the SAME worker can be in flight
only if the engine schedules them together; today it does not. So item 3 is:

* (a) a kernel entry that takes N slots and shares the weight loads across them, and
* (b) a lane/scheduler change that lets one worker hand two ready slots' units to the team
  as one work item, keeping the per-slot lead <= 1 quantum.

Doing (a) without (b) buys nothing (a per-slot loop around the existing kernel re-reads the
weights exactly as today). Doing (b) without (a) buys nothing either (two units run
back-to-back on the same team). The pair is the feature.

## 2. Kernel design (a)

Keep the existing single-slot entry untouched; add

```c
int  qwen_conv1d_int8_v2_multi_available(void);
void qwen_conv1d_int8_v2_multi_ctx(
        float *const *out, const float *const *in,
        const float *const *tail, const int *tail_cols, const float *const *residual,
        const int8_t *wq, const float *sw, const int32_t *wsum, const float *bias,
        int ch, int nslots, int length, int kernel, int dilation, int Cp);
```

Shape rules are the same as the single-slot entry (`Cp = cp(in_ch)`, any `in_ch`/`out_ch`
after the rectangular landing). The inner loop is restructured for the shared sweep:

* tile = 4 output channels x 2 positions (not 4x4): 8 `int32x4` accumulators per slot.
* with `S` slots: `8*S + 4` live vectors; `S <= 3` fits the 32 NEON registers without
  spills, `S = 2` is the comfortable product point (two ready slots per worker at cap >= 4).
* loop order: `for m0 { for t-tile { for kk { load w0..w3 once; for s in slots { load
  x0,x1 of slot s; a_s = dot(...) } } } }` — the weight vectors are loaded once per
  (m0, kk, k-step) and used by every slot, which is the entire win: the current kernel
  re-reads them per slot.
* the C array of accumulators spills gracefully; do not hand-unroll across slots.

Exactness: for one output element the sum over `k` visits the same products in the same
order as the single-slot kernel, and the epilogue expression is unchanged, so the batch is
bit-identical to running the single-slot kernel slot by slot. That is the oracle for the
self-test: `multi(n=2).out_s == single(in_s).out` bit for bit on the existing shape list
plus the two rectangular shapes.

## 3. Lane/scheduler design (b)

`qwen_tts.c` holds the per-slot unit state (`lane` field) and enqueues one unit per slot.
The batch needs a "cohort": when the worker's frame loop has >= 2 slots whose next decoder
quantum is ready, enqueue ONE unit carrying both slots' decoder states. The kernel call
sites are all in `causal_conv1d_blas` (and its convt/cnext siblings): each wants a single
`in`/`out`; the batched unit therefore needs a small per-slot view (pointers into each
slot's activation buffers), which is what `qwen_conv1d_int8_v2_multi_ctx` takes.

Acceptance for (b): lead <= 1 quantum per slot (no slot waits for another slot's decode),
and the unit time at B=2 slots is below 2x the single-slot unit time (that is the whole
point; a 2x cost means no sharing happened).

## 4. Metrics and gates

* Exactness: bit-identical outputs vs the single-slot path, per shape, in `--self-test`.
* Unit cost: `QWEN_SD_PHASE=1` `conv_up` ms per unit at 1 slot vs 2 slots on the lane
  (4 threads), same chunk=4, on the 16-core Neoverse-V2.
* Serving: C10 2x8, lane 4+elastic, `QWEN_SD_RES1_V2=1` + GLUE; STREAM p95 and
  `overlap share` against the same arm without the batch. WAVE screen first, then SOAK.
* Quality: paired WAV + mel-corr as for RES1_V2 (numerics are bit-identical, so this is a
  regression guard, not a negotiation).

## 5. Risks

* Register pressure: `8*S+4` vectors is the hard bound; do not raise the position tile
  back to 4 with S>1.
* The lane lead bound is the safety property; a cohort that waits for a second slot
  reintroduces exactly the blocking DL-1 removed.
* `tail`/`residual` are per slot; a batched unit must keep per-slot pointers, and the
  keep-the-residual-in-the-epilogue GLUE contract must stay per slot.
* Allocation: the batched unit must not allocate per call; per-slot scratch already exists.

## 6. Files/functions to touch

`qwen_tts_kernels.c` — `sd_dconv_worker`/`_ctx` (new multi entry, both leaves),
`qwen_tts_kernels.h`, the `#else` fallback section, `--self-test` (multi vs single oracle).
`qwen_tts_speech_decoder.c` — `causal_conv1d_blas` call sites, the per-slot views, the
`sd_wq` pack cache (weights already shared).
`qwen_tts.c` / `qwen_tts_thread.c` — cohort formation in the frame loop, lane submission.

## Next action

Implement (a) first and land it with the multi-vs-single self-test; it is exact and useful
on its own. Only then start (b), with the cohort gate behind a new default-off flag
(`QWEN_SD_MULTISLOT`, integer slot count) so the single-slot lane stays the control.
