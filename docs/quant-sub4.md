# Sub-4-bit quantization for the Code Predictor (E7) — survey, method, verdict

**Status: CLOSED 2026-07-14 — NO GO (see §5). E7.1 survey + E7.2/E7.3 measurements DONE;
E7.4 kernels never started, by design (quality gate failed first).**
Branch: `feat/quant-sub4`. Style: like `docs/pr17-review.md` — this doc is the durable
record, including a possible NO verdict (so nobody re-attempts sub-4 naïvely in 6 months).

---

## 1. Why re-open a closed question

PLAN.md §21.1 (2026-06-04) closed sub-int4 with: *"int2/int1 = research only; with simple
absmax/group scales quality dies; realistic floor = int4-CP / int8-Talker"*. The quant-ladder
measured it: teacher-forced per-codebook agreement vs bf16 on the 0.6B CP was **int8 78%,
int4 46% (collapsing to 23-27% on late codebooks c11-c15), q2 9%**.

That verdict was for **our naïve format**: Q4_0-style per-32 absmax round-to-nearest (RTN),
one fp16 scale per block, no error minimization, no activation awareness. The int4 story
itself teaches the parabola: int4 was "broken" (naïve), then "slow on M1" (unpack cost),
then became the **fastest 1.7B M1 config** once group scales + SDOT-native decode landed
(B1 + PR#17). The hypothesis of E7: the same parabola may apply one level down, because the
modern formats we never tried are precisely "the group-scale trick, one level up".

Why the CP is the right (and only) target:
- CP is the per-frame bottleneck (74-90% of frame time, 90.7% matvec) and its weights are
  re-read 16×/frame → on bandwidth-bound devices the win is linear in bits.
- q3 ≈ −24% bytes vs our q4 (4.5 bpw → 3.44), q2_k ≈ −42% (→ 2.625 bpw).
- The Talker is compute-bound and gates intelligibility (int4 Talker flips ~7% of words) —
  sub-4 there is pointless and dangerous. **Scope = CP only.**

Known headwind (must be beaten, not ignored): int4 already collapses on late codebooks, and
the decomposition experiments showed the drift lives in the **shared transformer** (int4
transformer + bf16 heads = 48%; bf16 transformer + int4 heads = 84%). So a sub-4 format must
be *better per-bit than Q4_0-RTN by a lot* on exactly the shared-transformer matvecs, or it
must go into a mixed map (sub-4 only on tolerant tensors).

## 2. Survey of modern formats (E7.1)

### 2.1 llama.cpp k-quants (the "scale-of-scale" family)

Superblocks of 256 weights, split into sub-blocks; sub-block scales (and mins) are
themselves quantized against one fp16 super-scale. Two-level scaling = fine-grained groups
at ~no byte cost. Quantization uses an **error-minimizing scale search** (not plain RTN):
for candidate scales, round, then closed-form least-squares rescale `s* = Σw·v·q / Σw·q²`,
pick the candidate minimizing weighted error Σw·(v − s*q)². Default weights w = v² (or the
imatrix when available).

| Format | Layout (per 256-superblock) | bpw | vs our q4 (4.5) |
|---|---|---|---|
| Q3_K | 16 sub-blocks×16; 6-bit signed sub-scales vs fp16 `d`; 3-bit quants q∈[-4,3] (2-bit planes + 1-bit hmask) | **3.4375** | **−23.6%** |
| Q2_K | 16 sub-blocks×16; 4-bit sub-scale + 4-bit sub-min vs fp16 `d`,`dmin`; 2-bit quants q∈[0,3], v = d·sc·q − dmin·m | **2.625** | **−41.7%** |
| Q4_K (ref) | 8 sub-blocks×32; 6-bit scales+mins | 4.5 | 0% (better quality than Q4_0 at same bpw) |

Key properties for us: sub-block size 16 (vs our 32) + asymmetric mins (Q2_K) + scale
search = exactly the tricks that rescued int4, one level up. **SDOT-compatible**: quants
unpack in-register to int8, integer dot per sub-block, one f32 fixup per sub-block.

### 2.2 IQ quants + imatrix (activation-aware, no QAT)

- **imatrix**: run the model on calibration text, accumulate mean x² per input channel of
  every matmul → use as weights `w` in the same scale search. Turns "minimize weight RMSE"
  into "minimize *output* error proxy". For us calibration = a few minutes of synth over
  varied text; no training.
- **IQ2_XXS/XS/S (2.06-2.5 bpw), IQ3_XXS/S (3.06-3.44 bpw)**: groups of 8 weights coded as
  an index into an E8-lattice codebook + packed signs. Best quality per bit known in ggml;
  the 2-bit tier is unusable without imatrix. Cost: codebook lookup in the inner loop
  (table-driven decode — slower and much more complex than bit-plane unpack; SDOT still
  possible after LUT expand).
- **IQ4_NL/XS**: non-linear 4-bit LUT — a *quality* upgrade at ~our current int4 cost;
  interesting fallback even if sub-4 fails.

### 2.3 AWQ (activation-aware channel scaling)

Protect the ~1% salient input channels (by mean |x|) by scaling W columns up before group
quant, folding s⁻¹ into the previous op. In our engine the fold point exists for q/k/v and
gate/up (the preceding RMSNorm weight), but NOT for down_proj (input = SwiGLU output) and
lm_heads (input = final norm... foldable, shared across 15 heads). Verdict: **a refinement,
not a format** — orthogonal to k-quants/IQ; try only if a format is borderline at the gate.

### 2.4 GPTQ / Hessian-based

Column-by-column quant with error feedback into remaining columns (needs per-tensor Hessian
from calibration). Strong at 3-4 bit, heavier machinery (solver, ordering). Deliberately
out of scope for v1: k-quants+imatrix reach most of the quality at a fraction of the
complexity, and our kernels want a block layout, not GPTQ's arbitrary residual structure.

### 2.5 Decision — candidates

1. **Candidate #1: Q3_K-style** (superblock 256, 16×16, 6-bit sub-scales, weighted scale
   search). Closest to our existing block kernels, cleanest SDOT story at 3 bit
   (2-bit plane + hmask → int8 in-register), −24% bytes.
2. **Candidate #2: Q2_K-style** (+ mins, asymmetric). The real prize (−42%) but the riskiest
   — the old q2 scored 9%.
3. **Candidate #3 (only if #1 is borderline): imatrix weighting** on top of #1/#2, then
   AWQ-style fold. IQ lattice codebooks only if everything else fails AND the byte math
   still justifies kernels (unlikely — decode cost).
4. **Strawmen for calibration of the experiment itself**: `q4_0` fake-quant (must ≈ the
   measured C int4 46% — validates the fake-quant harness), `q3_0`/`q2_0` naïve RTN
   (quantifies how much of the gap is *format* vs *bits*).

## 3. Method (E7.2): quality gate BEFORE any kernel

The old attempt's avoidable mistake was kernels-first. Here: **fake-quant in Python, zero C**.

- `tools/quant/fakequant_cp.py` quantizes→dequantizes the CP weights **that the C engine
  actually quantizes under `QWEN_CP_PREC`** (per layer: q/k/v/o + gate/up + down; plus the
  15 lm_heads; NOT codec_embeddings, NOT norms) and writes a modified `model.safetensors`
  (bf16). The engine then runs it as normal bf16 → the quantization error is baked into the
  weights, measured by the existing teacher-force harness with no engine changes.
- Rails: bf16 free run with `QWEN_DUMP_CODES` (`-j1 --temperature 0 --seed 42 -s ryan
  -l Italian`, fully deterministic). Replay: `QWEN_TF_CODES=<rails>` per variant model →
  per-codebook argmax agreement via `tests/quant_ladder.py`. Same instrument, same metric
  as the June quant-ladder → numbers directly comparable (int8 78% / int4 46% / q2 9%).
- Known small bias: dequantized values get re-rounded to bf16 in the safetensors (the real
  kernel would compute f32 from ints × fp16 scale). This *underestimates* format quality
  slightly → conservative, acceptable for a go/no-go gate.
- **Gate to proceed to E7.3/E7.4 kernels:** best sub-4 format on the full CP must land
  **well above int4's 46% — target zone ≥ 60-65% overall** with late codebooks not
  collapsing below ~40% (int4: 23-27%); otherwise mixed-precision per-tensor (E7.3) must
  recover ≥15-20% CP bytes vs pure int4 or the epic closes with NO.

## 4. Results (E7.2) — measured 2026-07-14

0.6B, 143 frames, `--seed 42 -s ryan -l Italian --temperature 0 -j1`, full protocol in
`samples/tests/2026-07-14_quant-sub4-ladder/` (`ladder_results.txt`). Harness sanity:
**bf16-TF control = 100.00%**, code0 identical across all dumps, and **q4_0 fake-quant
(46.67%) reproduces C int4 (46.34%)** → the fake-quant method is validated; the June and
today's ladders are directly comparable (int8 79.4% today vs 78% June — same instrument).

| Variant | bpw (CP matvec) | overall c1-15 | c1-c5 | c11-c15 | verdict |
|---|---|---|---|---|---|
| int8 (C) | 8.5 | **79.4%** | 80-85% | 69-83% | GOLD floor (June: 78%) |
| int4 (C) | 4.5 | **46.3%** | 53-68% | 26-36% | shipped baseline (June: 46%) |
| q4_0 fake-quant | 4.5 | **46.7%** | 50-69% | 26-34% | ✅ ≈ int4 C → method valid |
| q3_0 naïve RTN | 3.5 | **14.4%** | 15-44% | 4-7% | strawman |
| q3_k (candidate #1) | 3.4375 | **27.9%** | 29-55% | 14-19% | ❌ FAILS gate (needed ≥60%) |
| q2_k (candidate #2) | 2.625 | **4.9%** | 2-24% | 1-3% | ❌ dead |
| q2_0 naïve RTN | 2.5 | **1.3%** | 0-10% | 0-2% | strawman (June q2: 9%*) |

*June's q2 (9%) quantized only part of the CP (FFN hybrid knob); today's q2_0 covers all
matvec weights — hence lower. Not a regression, a different denominator.

**Reading:**
- **The format hypothesis was directionally right**: at equal bits, k-quant structure
  nearly *doubles* naïve RTN (q3_k 27.9% vs q3_0 14.4%; q2_k 4.9% vs q2_0 1.3%), matching
  the synthetic rel-RMSE ordering (q3_k 15.5% vs q3_0 23.0% weight error).
- **But the CP's quality cliff below 4.5 bpw is steeper than the format gain**: dropping
  4.5 → 3.44 bpw with the *better* format still loses 19 points (46.7 → 27.9), and late
  codebooks land at 14-19% (gate wanted ≥~40%). int4 was already at the edge of the cliff.
- Rough per-bit slope around the edge: ~18-19 agreement points per bpw between 3.4 and
  4.5 — to reach the ≥60% zone a 3.4-bpw format would need to behave like ~5.2 bpw.
  imatrix-class weighting (candidate #3) typically buys the equivalent of ~0.5 bit and our
  search already uses x² weighting (a weak proxy); it cannot plausibly bridge 28 → 60.
  Only QAT-class methods could — out of scope by design (no training in this engine's
  quant pipeline).

## 4b. Mixed-precision map (E7.3) — measured 2026-07-14

Same protocol (`ladder_heads_results.txt`). June's decomposition already showed the drift
lives in the shared transformer (int4 heads + bf16 transformer = 84%); today quantifies
how far DOWN the heads can go, and what the realistic mix buys:

| Variant | overall | late c11-c15 | CP bytes vs pure int4 |
|---|---|---|---|
| q3_k heads only (transformer bf16) | **77.0%** | 72-83% (flat!) | −7.0% |
| q2_k heads only (transformer bf16) | **57.2%** | 52-60% (flat) | −12.3% |
| mix: q4_0 transformer + q2_k heads | **38.8%** | 27-31% | −12.3% |
| pure int4 (ref) | 46.3% | 26-36% | 0% |

**Findings:**
- **The 15 lm_heads are extremely quant-tolerant**: q3_k heads sit at the int8 gold floor
  (77 vs 79.4), q2_k heads alone still beat FULL int4. And their per-codebook profile is
  FLAT — mechanistic confirmation that the late-codebook collapse is the *shared
  transformer accumulating error across the 15 intra-frame steps*, not head precision.
- **But heads are only ~29% of CP matvec bytes** (31.4M of 106.4M params) and ~4.3% of CP
  compute → the best defensible mix (q4 transformer + q2_k heads) saves **12.3% of CP
  bytes, below the 15-20% E7.3 gate**, while costing 7.5 agreement points vs pure int4
  (46.3 → 38.8). The tensor that would pay (the transformer) is exactly the one that can't.

## 4c. Follow-up round: BETTER 4-bit ("the legitimate heir") — measured 2026-07-14

Since sub-4 failed on the quality cliff, the same harness was pointed at the open cheap
question from §5: **better formats at the SAME 4.5 bpw**, plus the "pure int4+int4"
question (Talker int4 quality is what blocks it — int4 flips words). Three candidates,
all 4.5 bpw: `q4_0s` (identical Q4_0 layout, weighted scale search instead of absmax
RTN → **ships as a loader change, zero new kernels**), `iq4_nl` (non-linear 16-value
LUT), `q4_k` (superblock, asymmetric, two-level scales). Talker measured via
`QWEN_DUMP_CODE0` under TF (code0 = words; bf16 control 100%, fake q4_0 83.2% ≈ C int4
83.9% → method valid on the Talker too). Results (`ladder_q4_results.txt`,
`talker_code0_results.txt`):

| Format (4.5 bpw) | Talker code0 (words) | CP c1-15 | CP late c11-15 |
|---|---|---|---|
| int4 C today (absmax RTN) | 83.9% | 46.3% | 26-36% |
| **q4_0s** (search, same layout) | **90.2%** | 48.8% | 27-40% |
| q4_k | 84.6% | 52.9% | 34-44% |
| **iq4_nl** | 85.3% | **55.2%** | **36-52%** |
| int8 (reference) | ~96.9% (June) | 79.4% | 69-83% |

**The two components want different medicine** (mechanistically consistent):
- **Talker**: one wide-margin argmax per frame → what matters is SCALE accuracy.
  q4_0s alone cuts wrong-word frames by 42% (24 → 14 of 143). The fancy grids barely
  move it.
- **CP**: error compounds over 15 intra-frame steps → what matters is GRID accuracy.
  iq4_nl recovers +8.8 pts overall and +10-16 on the late codebooks (closes ~27% of the
  int4→int8 gap at identical bytes).
- Search breadth is nearly irrelevant on real weights: 1 candidate (signed absmax → -8)
  + closed-form weighted LSQ rescale ≈ full 19-candidate sweep (rel-RMSE 8.93 vs 8.87;
  naive 10.18) → the C port of q4_0s costs ~one extra multiply-accumulate pair per
  weight at load time. Fake-quant variant `q4_0s1` validates it end-to-end.
- iq4_nl kernel cost if ever wanted for the CP: kvalues fit int8 → NEON decode =
  nibble-unpack + `vqtbl1q` table lookup + SDOT (llama.cpp does exactly this; x86 =
  `vpshufb`). A real E7-style candidate — but gate it on ear need, not on this table.

**✅ q4_0s1 PORTED TO C (same day)** — `qwen_quantize_bf16_to_q4_0` in
`qwen_tts_kernels.c` now does signed-max→-8 + weighted-LSQ rescale (w=v²) in the same
single load pass; `QWEN_Q4_NAIVE=1` restores the old absmax RTN for A/B. Validation on
the binary: Talker code0 **90.91%** / CP **48.81%** (fake-quant predicted 92.31/48.86;
naive lever reproduces the historical 83.92/46.34 exactly), `--self-test` PASS,
`make test-golden` ALL PASS (bf16/int8 paths byte-identical, no int4 golden in the set).
Every `--int4` / quant-mixed config on every ISA gets the words-accuracy jump for free.
Ear A/B: `samples/tests/2026-07-14_quant-sub4-ladder/ear_int4_{OLD_naive,NEW_lsq}.wav`.

## 5. Verdict (E7.5) — **NO GO for sub-4-bit kernels** (2026-07-14)

- **E7.2 gate: FAILED.** Best modern format on the full CP: q3_k **27.9%** vs gate ≥60%
  (int4 baseline 46.3%, int8 gold 79.4%). q2_k 4.9% = dead. No q3/q2 SDOT kernels (E7.4
  not started, by design).
- **E7.3 gate: FAILED.** Max justifiable mixed saving = 12.3% CP bytes < 15-20% threshold,
  at a real quality cost.
- **What survives as durable knowledge:**
  1. The **format thesis is real but insufficient**: k-quant structure ≈ 2× naïve RTN at
     equal bits (27.9 vs 14.4; 4.9 vs 1.3) — the int4 parabola does NOT repeat because
     int4 was already at the edge of the CP's quality cliff (~18-19 agreement pts/bpw
     around 3.4-4.5 bpw). A 3.4-bpw format would need to act like ~5.2 bpw to pass; only
     QAT could do that (out of scope).
  2. **lm_heads tolerate 3-bit at the int8 floor** — useless as a speed lever, but a valid
     *memory* lever if a tiny-footprint 0.6B build is ever needed (heads q3_k: −7% CP
     bytes ≈ free quality-wise). Parked, not planned.
  3. The **fake-quant harness** (`tools/quant/fakequant_cp.py` + `run_sub4_ladder.sh`) is
     now a reusable instrument: any future format idea = one Python function + ~2 min of
     teacher-forced measurement, zero C. This is how sub-4 should have been tested the
     first time.
- **Do not re-attempt sub-4 on the CP without a fundamentally different method class**
  (QAT / distillation / architecture change). Post-training formats — including
  imatrix/AWQ refinements (~0.5-bit-equivalent gains) — cannot bridge 28% → 60%.
- Possible cheap follow-up OUTSIDE this epic's scope: **IQ4_NL-style non-linear 4-bit**
  as a *quality* upgrade at unchanged bytes (int4's 46% might improve at 4.5 bpw) — worth
  a single fake-quant run if int4 quality ever becomes the blocker.
