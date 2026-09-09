# C12-WIN forensic audit (2026-09-09, read-only, after the Codex ladder)

**Task.** Reconstruct repository state and experiment provenance after the C12-WIN session
was hard-stopped; verify the product runtime survived; separate idea-level from
implementation-level verdicts. **Question.** What do we actually know, what merely failed
to be implemented well, and what is the highest-confidence route to the missing C12 margin?
**Known facts.** Review base 181435e (HEAD 32f01d4 clean at review start); Codex commits
e8a9080, 5813540, e821628; profile `turin-c8a-32c-vnni-product`. **Unknowns.** Raw outputs
of the bf16/ConvT/glue microbenches (not in `.work/evidence`, box-only); overlap-conditioned
CP sections (instrumentation does not bucket them).

## P0 lineage
32f01d4 → 181435e (Fable review, DOCS) → e8a9080 TOOLING (`serving_profile.merge_profile_env`:
overrides accepted only for `parity.tunable_flags`, pinned values still refused; test added)
→ 5813540 EXPERIMENTAL DEFAULT-OFF (`QWEN_SD_BF16_PREUP`: persistent bf16 copies of the
decoder transformer + in/out proj, `qwen_matmat_bf16_packed` with bf16-packed activations,
f32 sgemm kept as the fallback; dispatch row `decoder.pre_up_bf16`; flag registered; also a
regenerated `qwen_flag_scope.h` scope table) → e821628 DOCS (six `.work` files, PLAN).
Worktree clean, no untracked files, reflog has no reset/stash in the session; the only stash
is an old one on `codex/amx-analysis`. `git diff 32f01d4..HEAD -- '*.c' '*.h'` is exactly
5813540. Mac build of HEAD + `--self-test` PASS; flag registry, flag parity, profile tests,
serving-profile tests, check_plan, repo integrity PASS.

## P1 product runtime
| mechanism | verdict | basis |
|---|---|---|
| DL-2 elastic lane (`QWEN_SD_LANE_SPLIT/ELASTIC`, mailbox, width cap) | PRESERVED | no hunk in qwen_tts.c/thread.c except one `unload` line |
| 4x8 / one worker per CCX, cap 4, q4, fail-fast | PRESERVED | profile JSON diff vs 32f01d4 = 0 lines; server untouched |
| RES1_V2 kernel, packer, self-test contract | PRESERVED | kernels.c diff = one registry string |
| VNNI Talker/CP, bf16 prefill | PRESERVED | untouched |
| first-chunk ramp 1,2,4 (qwen_tts.c:3536-3541) | PRESERVED (ramp 1,4 code removed as stated) | qwen_tts.c diff = +1 line |
| frozen product/control profiles | INTACT, not contaminated | 0-line diff; `QWEN_SD_BF16_PREUP` absent |
| strict preflight | ALTERED INTENTIONALLY (tunable overrides) | e8a9080, tested |
| `qwen_flag_scope.h` | ALTERED INTENTIONALLY (scope metadata for --effective-config) | flag_parity PASS |
Losing experimental runtime code left in tree: none (ConvT, split-input, alloc-only, ramp
were worktree-only and reverted). Default-off diagnostic left in tree: `QWEN_SD_BF16_PREUP`
(5813540) — inert unless requested; keep or drop is a housekeeping decision, not a risk.

## P2/P3 experiments (idea vs implementation)
| experiment | tested implementation | control / path / host | result | verdict |
|---|---|---|---|---|
| prefill helper | existing `QWEN_PREFILL_HELPER=1`, LOW-priority helper | frozen profile, 10-min C12 soaks, e1b1ec7 binary | short p95 0.966→0.915, pooled 0.923→0.915; TTFA p95 172→683, safe-start 417→922, stall@250 0.8-1.6 % | **mechanism VALIDATED** (inline admission = the short tail, ~0.05); **this implementation falsified** as production |
| fixed-B3 overlap diag | `1x8@0-7` cap 3, COST_MAP+STAGE, 5 long waves, 1349 active=3 iterations | diagnostic | overlap 54.9 % of wall, CP 22.2→36.5 (p95 22.7→37.6), Talker flat, unit 49.2 ms (conv_stack 43.3, transformer 5.2) | VALIDATED; CONFIRMS the review's model (predicted 55-60 %, 22, 35, 50) |
| pool spin | 4096/16384 vs 65536, 3 waves | frozen profile | Δ p95 ≤ +0.018, noise | IDEA FALSIFIED (as a lever) |
| bf16 pre-up | transformer + in/out proj only (88 MB f32 → 44 MB bf16), bf16-packed activations, per-call pack + transpose | frozen profile, 3-wave C12 screen; B3 diag confounded (39 % B1) | STREAM p95 0.842→0.825, short 0.843→0.829; paired mel-corr 0.9789 < 0.98; decoder ms unchanged | **THIS IMPLEMENTATION FALSIFIED** (quality, and it targeted the 5 ms piece); idea for the conv_stack f32 weights (convnext pw 67 MB, init conv 44 MB, convt 75 MB) weight-only bf16 / int8-weight-f32-accumulate UNTESTED |
| ConvT one-GEMM | zero-filled `[k·in_ch][full_len]` input panel + one f32 GEMM + full output materialisation | quantum bench, 4 threads, B1-B4, chunks 1-8 | bit-identical; +14-18 % (chunk 1-2), +26-32 % (chunk 4-8) | **THIS IMPLEMENTATION FALSIFIED**: the panel multiplies FLOPs/traffic by ~k/stride; the proposed geometry (weights `[in_ch][k·out_ch]`, one GEMM on the un-expanded input, col2im scatter, same FLOPs as the k sgemms) was not built. Direction: NOT dead |
| allocation-only glue | calloc→alloc for overwritten temporaries | quantum bench | no resolvable change (expected ≤1.5 ms ≈ 3 %, inside bench noise) | INCONCLUSIVE |
| RES1_V2 split-input | direct `tail`+`new` sources into the v2 quantise loop | quantum bench, exact parity | +0.3-2.0 % q4, +4.7-5.9 % q8 | THIS IMPLEMENTATION FALSIFIED; the memory saving (~95 MB/unit estimated) did not appear, so the estimate is suspect; the residual-epilogue + out-of-place snake pieces were NOT tested |
| ramp 1,4 | knob written, removed | — | not run | NOT TESTED (correctly recorded) |

ConvT direction dead? **NO**: (1) the tested kernel expanded the input by k, the proposal does
not; (2) f32 stayed — the 75 MB weight stream was never halved; (3) the per-tap sgemm
control already has M=out_ch, K=in_ch, N=in_len geometry that BLAS handles well, so the
win must come from fewer weight passes (one packed pass, bf16/int8), not from call count.

## P4 docs
Codex's six files are evidence-labelled and honest (MEASURED/DERIVED/HYPOTHESIS; ramp
recorded as cancelled; bf16 carries its quality caveat; XISA track present as a deferred
section). Corrections made (smallest): PLAN WIN-1 now states the helper CONFIRMS the
admission mechanism; PLAN WIN-2 reworded from "closed" to implementation-level verdicts with
the untested targets named; review §3 row 2 carries the cost-map split (transformer 5.2 ms,
not ~12). Unresolved: no raw evidence for the bf16/ConvT/glue microbenches on the Mac.

## P5 evidence vs the review
Overlap 54.9 % / CP 22.2→36.5 / unit 49.2: CONFIRM the causal model. Helper: does NOT
falsify "inline admission exists" — it demonstrates it (short −0.051) while showing the
LOW-priority helper is not a production solution (TTFA +511 ms). BF16: weakens one row of the
ranking (the transformer is ~5 ms, not ~12); the conv_stack f32 stream remains the target.

## P6 what survives (max 3)
1. **Admission off the critical path without delaying first audio** — evidence: helper arm
   short 0.966→0.915; alive because the failure was TTFA of the LOW-priority helper, not the
   mechanism; upside ~0.05 on short/pooled p95 (C12 short → ~0.91); risk medium (scheduling);
   smallest falsifier: helper at normal priority / `QWEN_PREFILL_CHUNK` interleaving on one
   10-min C12 soak, gate TTFA p95 ≤ 300 ms and short p95 ≤ 0.92.
2. **Decoder residency via the conv_stack f32 weight stream** (convnext pw, initial conv,
   convt: ~190 MB/unit) with weight-only bf16 or int8-weight/f32-accumulate, and ConvT as
   one un-expanded GEMM — evidence: unit 49.2 ms is 43.3 conv_stack, tax scales with unit
   time, the tested arms never touched these weights; upside 6-10 ms/unit (→ overlap share
   55 → ~45 %, ≈ −0.03..−0.05 STREAM at B3); risk: quality gate (paired bank), medium;
   smallest falsifier: microbench convnext pw + initial conv in weight-only bf16 at M=4-22,
   then CP-in-overlap at pinned B3.
3. **Combined residual-unit glue** (residual in the v2 epilogue, out-of-place snake, no
   calloc, split input together) — evidence: SDUP resadd 3.5 + alloc 1.5 ms; separate
   pieces were neutral within bench noise; upside 3-5 ms/unit; risk low; falsifier: one
   microbench of the combined change, promote only if ≥ 3 ms.
The architecture review stands; today's evidence changes only the cost attribution inside
the unit (transformer 5 ms) and upgrades admission from "inferred" to "measured".
