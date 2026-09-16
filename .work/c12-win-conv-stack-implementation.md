# C12-WIN-11 — Decoder conv_stack: ConvT one-GEMM (un-expanded) + fused epilogue, then bf16 weights

**Task.** Implementation spec. **Question.** How to cut the f32 weight/activation traffic of
the decoder's conv_stack (43.3 of the 49.2 ms unit) without multiplying arithmetic and
without bf16 activations. **Known facts.** Per 4-frame unit on 4 lane threads (DL-4 SDUP +
cost map): res1 17.1, convt 8.6, res2 4.0, resadd 3.5, alloc 1.5, final 1.3, snake 1.0,
transformer 5.2, convnext+initial conv ≈ 7. ConvT weights: block 0 `1536→768, k=16` =
75 MB f32 (the whole other three blocks 15 MB); convnext pw 67 MB; initial conv 44 MB. Codex's
one-GEMM used a zero-expanded `[k·in_ch][full_len]` panel and lost 14-32 %; the bf16 arm
used bf16 ACTIVATIONS on the transformer and failed mel 0.9789. **Unknowns.** How much of
convt's 8.6 ms is the scalar scatter + `full` calloc vs the GEMMs (SD_PHASE splits it:
`_cg/_cm/_cs` counters exist at qwen_tts_speech_decoder.c:911-931 — read them first).

## 1. OBJECTIVE
Reduce decoder-lane residency by >= 4 ms/unit from the ConvT path (target 8.6 → <= 4.5 ms)
in two steps: (A) exact-parity restructure — one un-expanded GEMM per ConvT layer with the
tap accumulation, carry and bias fused into a final-form epilogue; (B) weight-only bf16
(f32 activations, f32 accumulate) through a custom kernel for the bandwidth-bound layers
(ConvT block 0, then convnext pw, initial conv). Not optimized: res1/res2 (V2 stays), the
transformer, snake, quantum, lane policy, topology.

## 2. WHY THIS SHOULD WORK
FACT: `causal_conv_transpose1d_blas` (qwen_tts_speech_decoder.c:905-936) runs `kernel`
sgemms (16 for block 0), each `M=out_ch, N=in_len, K=in_ch` into `rk`, then a scalar
scatter `dst[t*stride+k] += src[t]` with a bounds test per element; `cs_convt` (2326)
allocates `full` with calloc (`out_ch × full_len`), then a second pass adds carry and bias
into `out` and saves the new carry. FACT: block 0 has N = in_len = 16, so its 75 MB of
weights are streamed for 16 columns — pure weight bandwidth (~2 ms at 4 threads); blocks
1-3 are small-weight, large-N: their cost is the scatter/copy passes (block 3: 96×7680 f32
= 2.9 MB per pass, four passes). INFERENCE: (A) removes `rk`, `full`, the calloc memset,
the bounds-checked scatter and the carry/bias pass, replacing them with one write of the
final output — ~2-3 ms; (B) halves block 0's weight stream — ~1 ms — and halves the
convnext/initial-conv streams (111 MB → 55) — ~2 ms; total 5-6 ms/unit, i.e. overlap share
55 → ~49 % at B3 ≈ −0.02..−0.03 STREAM. HYPOTHESIS: the CP-in-overlap tax per iteration also
drops if the interferer is the weight stream (falsifier in §11).

## 3. EXACT CURRENT DATAFLOW (per ConvT layer, per unit)
Input `in[in_ch][len]` f32 (row-major, len = 16/128/640/2560 for blocks 0-3), weights packed
at load by `sd_pack_convt` (131) as `p[(k*in_ch + ic)*out_ch + oc]`, stride r = 8/5/4/3,
kernel k = 2r, `carry[out_ch][k−r]` per block in the stream state (`cs_up_carry[b]`).
`cs_convt` (2326): `full = calloc(out_ch × full_len)`, full_len = (len−1)·r + k;
`causal_conv_transpose1d(full, in, ...)` → for each tap k: `SD_GEMM(Trans, NoTrans, out_ch,
len, in_ch, wk[in_ch][out_ch], in)` → `rk[out_ch][len]`; scatter `full[oc][t·r + k] += rk[oc][t]`
(bounded by full_len). Then `out = alloc(out_ch × out_len)`, out_len = len·r; per oc:
`full[0..k−r) += carry`; `out[t] = full[t] + bias` for t < out_len; `carry = full[out_len ..
out_len + k−r)`. All on the lane team (4 threads) via `SD_GEMM`/BLAS partitioning; scratch
from the per-stream arena (`sd_tmp_alloc`, never freed until reset).

## 4. INTENDED NEW DATAFLOW
Mathematics (k = 2r): output position o = t·r + j with j = o mod r, t = o div r receives
exactly two taps: tap j from input column t and tap j+r from input column t−1 (t−1 < 0 →
the previous unit's last column = the carry). Therefore:

```
load time (once, per ConvT layer):  Wstack[(j*out_ch + oc) * in_ch + ic] = w[ic][oc][j]   # [k·out_ch][in_ch], row-major
                                    (step B: same layout in bf16)
per unit:
  R[k·out_ch][len] = Wstack[k·out_ch][in_ch] × in[in_ch][len]        # ONE GEMM, M=k·out_ch, N=len, K=in_ch
                                                                       # FLOPs identical to the k sgemms; input NOT expanded
  epilogue (direct, final form, per oc, parallel over oc on the lane team):
    for t in 0..len-1:
      for j in 0..r-1:
        a = R[(j)*out_ch + oc][t]
        b = (t >= 1) ? R[(j+r)*out_ch + oc][t-1] : carry[oc][j]
        out[oc][t*r + j] = a + b + bias[oc]
    for j in 0..r-1: carry[oc][j] = R[(j+r)*out_ch + oc][len-1]      # new carry, k−r == r entries
```
Buffers: `R` (k·out_ch·len f32) is the only scratch: block 0 0.8 MB, block 3 5.9 MB (vs
today's `full` 3 MB + `rk` 1 MB) — acceptable in the arena; `out` is written once, final
form; no `full`, no calloc, no scatter pass, no carry/bias pass. Step B replaces the BLAS
call for block 0 (and later convnext pw / initial conv) with:

```
qwen_gemm_w16f32(R, Wbf16 [M][K], X f32 [K][N], M, N, K, ldx=N, ldr=N)   # lane team, rows split over threads
  per thread: rows m0..m1 of M; per row: for n-panel of 16 columns (N <= 16 → one panel):
     acc = 0 (zmm); for kk in 0..K-1: w = bf16→f32 (shift left 16 bits, exact), acc += w * X[kk][n0:n0+16]
     store R[m][n0:n0+16] = acc
  X panel (K×16 f32 = 96 KB for K=1536) stays in L2; weights streamed once per panel.
  N > 16 (blocks 1-3, initial conv N=22, convnext N=8/16): loop panels; weights re-read from L2/L3.
```
bf16→f32 expansion is exact; activations stay f32; accumulation f32 (`_mm512_fmadd_ps`).
Weight rounding happens ONCE at load (round-to-nearest-even). No activation packing.

## 5. FORBIDDEN IMPLEMENTATIONS
* DO NOT expand the input into a `[k·in_ch][full_len]` (or any zero-filled) panel; DO NOT
  express ConvT as a direct conv over an upsampled/zero-stuffed input. Arithmetic must stay
  k·out_ch·in_ch·len MACs per layer, exactly as the k sgemms.
* DO NOT keep `full`/calloc or add another full-size intermediate after `R`.
* DO NOT repack weights per call; `Wstack` (f32 and bf16) is built in `qwen_speech_decoder_load`.
* DO NOT pack activations to bf16 (`qwen_bf16_pack_rows`) or use `qwen_matmat_bf16_packed`
  for any decoder GEMM — that is the failed C12-WIN-1a arm; only weights may be bf16.
* DO NOT use int8 anywhere in this item.
* DO NOT touch res1/res2 (`qwen_conv1d_int8_v2`) or the transformer.
* DO NOT change stream state layout (`cs_up_carry` stays `[out_ch][k−r]`).

## 6. IMPLEMENTATION SKETCH
```
/* qwen_tts_speech_decoder.c */
static float    *sd_pack_convt_stack(const float *w, int in_ch, int out_ch, int kernel);   /* [k*out_ch][in_ch] f32 */
static uint16_t *sd_pack_convt_stack_bf16(const float *stack, size_t n);                    /* step B */
/* load: alongside the existing sd_pack_convt() at :1030/:1038, keep the old pack for the control path */
static float *cs_convt_stack(const float *in, int in_ch, int out_ch, int len, int kernel, int stride,
                             const float *wstack, const uint16_t *wstack16, const float *b, float *carry) {
    int M = kernel * out_ch, N = len, K = in_ch;
    float *R = sd_tmp_alloc(M * N * 4);
    if (wstack16 && sd_convt_w16_enabled()) qwen_gemm_w16f32(R, wstack16, in, M, N, K, N, N);
    else SD_GEMM(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, wstack, K, in, N, 0.0f, R, N);
    float *out = sd_tmp_alloc(out_ch * len * stride * 4);
    /* epilogue of §4, dispatched over oc on the lane team via sd_pool_run (same as the V2 conv) */
    sd_tmp_free(R); return out;
}
/* cs_convt(): if (sd_convt_stack_enabled()) return cs_convt_stack(...); else existing path (control). */
/* qwen_tts_kernels.c: void qwen_gemm_w16f32(float *R, const uint16_t *W, const float *X, int M, int N, int K, int ldx, int ldr);
   AVX-512 only; generic fallback = expand row to f32 then dot (correctness reference for the self-test). */
```
Flags: `QWEN_SD_CONVT_STACK=1` (step A), `QWEN_SD_CONVT_W16=1` (step B, requires A);
register both; dispatch rows `decoder.convt_stack`, `decoder.convt_w16`. Later call sites
for step B (separate commits, same kernel, same gates): `convnext_mlp` pw1/pw2
(qwen_tts_speech_decoder.c ~2465-2495, weights `[4096][1024]`/`[1024][4096]`), the initial
conv `cs_conv1d` k=7 1024→1536 (its f32 im2col path).

## 7. FILES / FUNCTIONS TO TOUCH
* `qwen_tts_speech_decoder.c`: load-time packing (next to `sd_pack_convt`), `cs_convt`
  gate, new `cs_convt_stack`, free in the decoder unload; SDUP counters `sd_up_convt` keep
  measuring the whole call.
* `qwen_tts_kernels.c/.h`: `qwen_gemm_w16f32` + self-test cases.
* Flag registry, `qwen_flag_scope.h`, dispatch rows. Tests: self-test (§9).
Do NOT touch: `causal_conv_transpose1d_blas` (control), `cs_convt_direct`, res convs, V2
kernel, lane, qwen_tts.c, profiles, transformer, `qwen_matmat_bf16*`.

## 8. NUMERICAL / SEMANTIC INVARIANTS
* Step A: same taps, same inputs; summation order changes (a + b + bias vs sequential
  accumulation into `full`) → bounded, max_abs <= 1e-5 × max|out|, typical 1e-7; duration
  identical; carry values identical within the same bound; stream continuity exact in
  structure (carry contract unchanged).
* Step B: weight rounding to bf16 once; activations and accumulation f32. Expected paired
  mel-corr >= 0.995 (the RES1_V2 change measured 0.995-0.998 with a far larger perturbation);
  gate 0.99 + ear as for any numerical change (ENGINEERING §12). NOT exact.
* No change to quantum, lane, cadence, topology, request semantics, profiles.

## 9. LOCAL CORRECTNESS ORACLE
1. Self-test cases: (a) `convt_stack` vs `causal_conv_transpose1d_blas` on the four block
   shapes with random input, carry and bias, two consecutive units (continuity): max_abs
   <= 1e-5 relative; (b) `qwen_gemm_w16f32` vs scalar f32 reference on the bf16-rounded
   weights: EXACT (same rounding, same f32 FMA order per row must be reproduced by the
   reference: accumulate in k order).
2. `tests/decoder_batch_parity` (the binary Codex used) with `QWEN_SD_CONVT_STACK=1`: PASS
   within 1e-5 (not bit-identical).
3. Paired audio A/B for step B (`QWEN_SD_CONVT_W16=1` vs control): the Phase-B bank recipe
   (`turin-…-control` style two-profile run or CLI temp 0 seed 42 on 10 texts):
   `compare_audio.py` mel-corr >= 0.99 every pair, duration identical, wav_qc equal.
4. ASan build of the self-test; flag registry; profile validation.

## 10. MICROBENCH DESIGN
Turin host, `qwen_tts_decode_quantum` (existing entry), 1.7B, 4 decoder threads pinned
`taskset -c 4-7`, B1 and B3, chunks 4 and 8, 5 warm repetitions, p50 wall; plus one run
with `QWEN_SD_PHASE=1` to read `[SDUP] convt=` per unit. Control: flags off. GO to step B
only if step A cuts the B1/chunk-4 call by >= 2.0 ms (noise floor ~0.8 ms = 1.5 %) with
parity; GO to the server A/B only if A+B together cut >= 4.0 ms on B1/chunk-4 AND the
`convt=` phase is <= 4.5 ms. Convnext/initial-conv call sites: each must show >= 1.0 ms on
its own before being kept.

## 11. SERVER A/B GATE
Single CCX `1x8@0-7`, cap 4, 1.7B, long bank, `QWEN_STAGE_TRACE=1`, 5 waves, control vs
treatment (two profiles differing only in the two flags, as for RES1_V2): decoder unit
mean (SDPHASE total at frames=4) must drop >= 4 ms; overlap share must drop; report CP ms in
overlap (if it drops by >= 3 ms the weight-stream hypothesis is supported). Then C12 10-min
soak under the frozen profile + flags: pooled/short/conversational p95 vs the C12-WIN-10 gate
numbers; kill on any prebuffer/safe-start/stall regression or any quality failure.

## 12. STOP / REVERT RULES
STOP after step A microbench if the gain is < 1 ms or parity fails or `R` had to be
materialised twice. STOP after step B if the kernel is slower than BLAS at N=16 (then the
kernel, not the idea, is wrong — one allowed correction: 2-row register blocking of the
weight rows), or the paired mel-corr < 0.99. REVERT immediately if any block shape is slower
than the control at B1-B4 chunk 4, or the treatment needs the product profile changed to
run. Never "fix" a slow result by expanding the input or packing activations.

## 13. SUCCESS STATE
Two commits: "decoder: ConvT as one un-expanded GEMM with a fused epilogue
(QWEN_SD_CONVT_STACK)" and "decoder: weight-only bf16 GEMM for the bandwidth-bound conv
layers (QWEN_SD_CONVT_W16)"; evidence `.work/c12-win-conv-stack-<date>.md` with the SDUP
before/after table, parity outputs, the single-CCX A/B and the soak; PLAN C12-WIN-11
closed; flags stay default-off until the C12-WIN-8 qualification promotes them into the
Turin profile (step A may be promoted on parity alone; step B needs the ear check).
