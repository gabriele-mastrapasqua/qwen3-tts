# C12-WIN-12 — Residual-unit glue removal on the VNNI per-item decoder (one combined change)

**Task.** Implementation spec. **Question.** Can the residual unit (snake1 → res1 k=7 →
snake2 → res2 1x1 → residual add), executed 12 times per unit, lose its copy/memset passes
without moving the cost into the V2 quantisation loop? **Known facts.** SDUP per unit:
resadd 3.5 ms, alloc 1.5 ms; inside res1's 17.1 ms the wrapper builds `ext` = [tail|in],
callocs `full` (out_ch × ext_len), computes ext_len outputs and copies the last `len`
columns out (`cut`) — qwen_tts_speech_decoder.c:2151-2247. Codex's isolated variants
(calloc→alloc; split input) were neutral or 0.3-6 % slower; the residual-in-epilogue and
out-of-place-snake pieces were never tested. **Unknowns.** The exact share of the ext/full/cut
passes (SD_PHASE `sd_c1_ext/cut` counters exist — read them first).

## 1. OBJECTIVE
Remove five full-size activation passes per residual unit (memcpy of the residual, calloc
memset of `c2_out`, the separate residual-add loop, the `ext` build and the `cut` copy) by
ownership transfer and epilogue fusion, keeping the V2 kernel's contiguous per-position
quantisation loop untouched. Target >= 3 ms per 4-frame unit on 4 threads at B1/B3 q4
(expected 4-6). Not optimized: kernel arithmetic, quantisation contract, res1 tap loop,
convt, snake math, quantum, lane.

## 2. WHY THIS SHOULD WORK
FACT: per residual unit at block 3 (96 × 7680 f32 = 2.9 MB) the code performs: `res =
memcpy(signal)` (1 read + 1 write), `ext` build (1 read + 1 write of in_ch × (len+6·dil)),
`full` calloc (1 write), `cut` (1 read + 1 write), `c2_out` calloc (1 write), add loop (2
reads + 1 write): ~10 passes ≈ 29 MB of traffic per unit at block 3, ~15 MB at block 2 —
against ~4.5 ms of actual conv work per block-3 unit. INFERENCE: on 4 threads with ~30 GB/s
effective, 29 MB ≈ 1 ms per residual unit at block 3, ~0.5 at block 2 → 3 × 1.5 ≈ 4.5 ms per
4-frame unit plus the calloc memsets. Codex's split-input arm did not show it because the
`ext` build was replaced by per-row branching inside the kernel's row loop (INFERENCE from
its +5 % at q8) — the spec below keeps ONE pointer/stride selection per input position and
zero changes to the channel loop. HYPOTHESIS: combined, >= 3 ms/unit, exact parity.

## 3. EXACT CURRENT DATAFLOW (per residual unit r of up-block b; qwen_tts_speech_decoder.c ~2500-2575)
1. `res = sd_tmp_alloc(ch×len); memcpy(res, signal)` — the residual copy.
2. `snake_activation(signal)` in place (snake1).
3. `c1_out = cs_conv1d(signal, ch, ch, len, 7, dil, w1, b1, cs_res_tail[b][r], warm)`:
   warm path (2151): `ext = alloc(ch × ext_len)`, per channel memcpy tail (6·dil cols) then
   in (len cols); `full = calloc(ch × ext_len)`; `causal_conv1d(full, ext, ...)` → V2 kernel
   computes ALL ext_len positions (the first 6·dil are garbage-context outputs); `out =
   alloc(ch×len)`; per channel memcpy `full[.., tail_cols..]` → out (`cut`); tail saved via
   `cs_save_tail` (last 6·dil columns of `in`, cheap). Free ext, full.
4. `snake_activation(c1_out)` in place (snake2).
5. `c2_out = sd_tmp_calloc(ch × len)`; `causal_conv1d(c2_out, c1_out, w2, b2, k=1)` → V2
   kernel (kernel=1) writes every element.
6. `signal[i] = res[i] + c2_out[i]` loop; free c2_out, res. (`cs_conv1d_fused_residual` is
   AMX-D only: inert here.)
V2 kernel (`sd_dconv_worker`, qwen_tts_kernels.c:10049-10140): per time block, the row-fill
loop quantises rows r = t0−pad .. t1+3 reading `j->in[ic*L + p]` (a channel-strided gather
per position, unavoidable — the input is [ch][len]), rows with p < 0 or p >= L become
u8 0x80 with scale 0; the tile loop then reads contiguous u8 rows; epilogue writes
`o[jj] = reduce(facc) − corr + bias`.

## 4. INTENDED NEW DATAFLOW
```
per residual unit (signal is the residual and stays untouched until the epilogue):
  act  = sd_tmp_alloc(ch×len)                      # NEW: snake1 out-of-place: act = snake(signal)
  c1   = sd_tmp_alloc(ch×len)                      # plain alloc, no calloc
  conv1d_v2_ctx(out=c1, in=act, len, tail=cs_res_tail[b][r], tail_cols=6·dil, ...)   # NO ext, NO full, NO cut
       # kernel row-fill: for row position p in [t0−pad, t1+3]:
       #   src = (p < 0) ? (tail + (tail_cols + p)), stride tail_cols   # left context row
       #       : (p < L) ? (in + p),                stride L            # new input row
       #       : zero row (as today)
       #   -- ONE (src, stride) choice per ROW; the ic loop is unchanged: frow[ic] = src[ic*stride]
       # time loop covers only t in [0, L) of the NEW positions → out[m*L + t], final form
  cs_save_tail(tail, act, ch, len, tail_cols)      # as today (context = the conv1 INPUT, i.e. act)
  snake2(c1) in place                              # unchanged
  c2   = sd_tmp_alloc(ch×len)                      # plain alloc
  conv1d_v2_ctx(out=c2, in=c1, kernel=1, ..., residual=signal)   # NEW: epilogue o[jj] = (acc − corr + bias) + residual[m*L + t + jj]
  free(act); free(c1); free(signal); signal = c2   # ownership transfer; no add loop
```
Kernel contract: `qwen_conv1d_int8_v2` gains two optional inputs — `(tail, tail_cols)` for
the left context (NULL → today's behaviour: positions < 0 are zero rows, exactly the cold
path) and `residual` (NULL → no add). Quantisation per position is unchanged (same amax over
the same channel values, whether the row comes from `tail` or `in`), so outputs for t >= 0
are bit-identical to the ext path (the continuation self-test already asserts this
contract). Memory: `ext`, `full`, `res` disappear; `act` replaces `res` (same size); `c2`
replaces `c2_out`. Passes removed per unit: memcpy(res), ext build, calloc(full), cut,
calloc(c2_out), add loop → 6 of ~10. Also: the kernel no longer computes the 6·dil garbage
output columns (37 % of the initial conv's outputs are not affected here — that conv is
f32 BLAS, out of scope).

## 5. FORBIDDEN IMPLEMENTATIONS
* NO branch per element or per channel inside the row-fill `ic` loop or the tile loop: the
  (src, stride) pair is chosen once per row, outside the channel loop.
* NO new contiguous [tail|in] copy anywhere (in the wrapper or inside the kernel) — that is
  the `ext` you are removing.
* NO in-place snake1 on `signal` (it is the residual now); NO memcpy to preserve it.
* NO change to per-position quantisation, scales, weight layout or tap order of V2.
* NO calloc→alloc where the consumer does not write every element: `out` of `cs_conv1d`
  cold path and the non-V2 fallback paths keep their existing allocation calls.
* NO residual add via a separate pass "for now"; the epilogue add is the mechanism.
* NO change to `cs_save_tail` semantics (tail = last 6·dil columns of the conv1 input).
* Do not touch the AMX-D / Design-D / KleidiAI paths; VNNI per-item only.

## 6. IMPLEMENTATION SKETCH
```
/* qwen_tts_kernels.h/.c — extend the job, keep the old entry as a wrapper */
typedef struct { ...existing sd_dconv_job_t fields...,
                 const float *tail; int tail_cols;      /* left context [ch][tail_cols] or NULL */
                 const float *residual; } sd_dconv_job_t;
void qwen_conv1d_int8_v2_ctx(float *out, const float *in, const float *tail, int tail_cols,
                             const float *residual, const int8_t *wq, const float *sw,
                             const int32_t *wsum, const float *bias,
                             int ch, int length, int kernel, int dilation, int Cp);
void qwen_conv1d_int8_v2(...)  { qwen_conv1d_int8_v2_ctx(out, in, NULL, 0, NULL, ...); }   /* unchanged contract */

/* sd_dconv_worker row-fill (replace the `if (p < 0 || p >= L) { zero row }` with): */
const float *src; int sstride;
if (p >= 0 && p < L)            { src = j->in + p;                    sstride = L; }
else if (p < 0 && j->tail && -p <= j->tail_cols) { src = j->tail + (j->tail_cols + p); sstride = j->tail_cols; }
else                            { memset(qr, 0x80, Cp); sc[r] = 0.0f; continue; }
for (ic = 0; ic < ch; ic++) { float v = src[(size_t)ic * sstride]; frow[ic] = v; amax = max(amax, |v|); }
/* everything after (scale, SIMD quantise, tile loop) byte-identical */
/* epilogue: */ o[jj] = _mm512_reduce_add_ps(facc[i*4+jj]) - corr[i][jj] + bv + (j->residual ? j->residual[(size_t)(m0+i)*L + t + jj] : 0.0f);

/* qwen_tts_speech_decoder.c */
static void snake_activation_oop(float *dst, const float *src, int ch, int len, const float *alpha, const float *beta);
/* = qwen_snake_activation with a separate destination; same math, same SIMD, same threading */
static float *cs_conv1d_v2_ctx(const float *in, int ch, int len, int kernel, int dil, const float *w, const float *b,
                               float *tail, int warm, const float *residual);   /* V2-only warm/cold wrapper: alloc out, call kernel, save tail */
/* residual-unit loop: use the §4 sequence when sd_res1_v2_enabled() && QWEN_SD_GLUE=1; else the existing code (control) */
```
Flag `QWEN_SD_GLUE=1` (default off, registered, dispatch row `decoder.glue_fused`), requires
`QWEN_SD_RES1_V2=1`; cold first unit (warm==0) uses `tail=NULL`.

## 7. FILES / FUNCTIONS TO TOUCH
* `qwen_tts_kernels.c`: `sd_dconv_job_t`, `sd_dconv_worker` row-fill + epilogue,
  `qwen_conv1d_int8_v2_ctx`, self-test cases; `qwen_tts_kernels.h`.
* `qwen_tts_speech_decoder.c`: residual-unit loop (~2500-2575), `snake_activation_oop`,
  `cs_conv1d_v2_ctx`; leave `cs_conv1d` (control) intact.
* Flag registry, `qwen_flag_scope.h`, dispatch row.
Do NOT touch: `qwen_conv1d_int8_v2_pack`, the tile loop, `cs_convt`, `cs_save_tail`,
transformer, qwen_tts.c, profiles, AMX/Arm paths.

## 8. NUMERICAL / SEMANTIC INVARIANTS
* EXACT (bit-identical) audio vs `QWEN_SD_RES1_V2=1` control: same quantised rows, same
  tile arithmetic, `(x + bias) + res` vs `res + (x + bias)` is the same float addition;
  snake out-of-place computes the same values. Verify with sha256 of the WAV at temperature
  0 (the golden harness's md5 caveat does not apply within one binary/ISA).
* Stream continuity unchanged (tail = last 6·dil columns of the conv1 input).
* No change to quantum, lane, topology, profiles, request semantics.

## 9. LOCAL CORRECTNESS ORACLE
1. Self-test: for each existing `conv1d_int8_v2` case add (a) `v2_ctx(tail, in)` vs `v2(ext)`
   restricted to t >= 0: EXACT (max_abs = 0); (b) residual epilogue vs separate add: EXACT.
2. CLI temperature 0 seed 42, five bank texts, 1.7B and 0.6B: WAV sha256 identical between
   `QWEN_SD_GLUE=0/1` (both with `QWEN_SD_RES1_V2=1`).
3. ASan build of self-test + one CLI decode; flag registry; `check_plan`.

## 10. MICROBENCH DESIGN
Turin, `qwen_tts_decode_quantum`, 1.7B, 4 threads `taskset -c 4-7`, B1/B3/B4, chunk 4 (and
8 for information), 5 warm repetitions, p50, plus `QWEN_SD_PHASE=1` for `[SDUP] res1=,
res2=, resadd=, alloc=` and `sd_c1_ext/cut`. Control `QWEN_SD_RES1_V2=1`, treatment
`+QWEN_SD_GLUE=1`. GO to the server A/B only if chunk-4 p50 improves >= 3.0 ms at B1 AND at
B3 (noise floor ~0.8 ms). Report the per-piece counters, not only the total.

## 11. SERVER A/B GATE
Single CCX `1x8@0-7`, cap 4, long bank, STAGE trace, 5 waves: decoder unit −3 ms or better,
overlap share down, CP-in-overlap reported. Then the C12 10-min soak with the frozen profile
+ flag: no regression on prebuffer/safe-start/stall, pooled/short/conversational p95 reported
against the C12-WIN-10/11 numbers.

## 12. STOP / REVERT RULES
STOP after the microbench if < 2 ms, or if any piece is slower (the counters say which;
one allowed correction: if `tail` rows are the slow part, prefetch the tail row block once
per time block — still no per-element branch). REVERT immediately on any non-zero max_abs,
any WAV sha mismatch, any slower B, or if the change needs the profile edited.

## 13. SUCCESS STATE
Commit "decoder: fuse the residual unit's context, epilogue and ownership on the VNNI path
(QWEN_SD_GLUE)"; evidence `.work/c12-win-vnni-glue-<date>.md` (SDUP before/after, parity
sha list, microbench table, A/B); PLAN C12-WIN-12 closed; the flag can be promoted into the
Turin profile on exact parity + the microbench/A-B gates (no ear check needed: bit-identical).
