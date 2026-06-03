# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-06-01

Core engine is **COMPLETE** and producing good audio for both 0.6B and 1.7B.
INT8 is validated end-to-end this session. The current focus is **hybrid
mixed-precision quantization** + making the engine actually fast off Apple Silicon
(cross-OS threading, x86 SIMD). History is compacted at the bottom — nothing dropped.

---

## CURRENT FOCUS — Phase 21: Hybrid quant + cross-OS threading + x86

> The thesis (à la antirez's DS4 mixed-quant engine, applied to OUR pipeline): we own
> the engine, zero deps → assign bits **per-component / per-tensor** by two axes —
> (1) how much that tensor costs in bandwidth/time, (2) how much quant noise the output
> tolerates in the flow. Quantize hard where it's *heavy AND tolerant*; keep precision
> where errors cascade. Not uniform quant — target the bottleneck.

### 21.0 Validated this session (baseline for everything below)

- **INT8 end-to-end** (Talker 1.7B + CP both models, preset + WDELTA `.qvoice`):
  - 0.6B RTF 1.70 → **1.29 (−24%)** (CP-only int8; Talker stays bf16 on 0.6B, see below)
  - 1.7B RTF 2.66 → **1.79 (−33%)** (Talker 65.8→49.7, CP ~64→59.2 ms/f)
  - Quality validated by ear (preset, custom voice, streaming, server). EOS normal.
  - Fixes that unblocked it: (1) `qwen_ftz_on()` FTZ for int8-induced denormals (FPCR
    bit24 ARM / MXCSR FTZ+DAZ x86); (2) route `qwen_matvec_int8_qkv` through the fused
    per-q/k/v path (inline GCD block hung at 4 threads); (3) drop the CP `cp_h>=2048`
    gate; (4) re-quantize after a WDELTA voice override (was using stale CV weights);
    (5) batched int8 prefill (was forced sequential).
- **Streaming TTFA 1571 → 560 ms (−64%)**: wired `--stream-chunk` (was ignored), ramped
  first chunk (2 frames then configured size), batched int8 prefill (477→226 ms).
- **Facts corrected (believed wrong for months):**
  - The Code Predictor is **hidden=1024 on BOTH 0.6B and 1.7B** (verified via tensor
    shapes). Only the Talker differs (0.6B 1024 / 1.7B 2048). → CP quant helps both equally.
  - **INT4 is slower than INT8 on the Talker** (64.3 vs 45.9 ms/f, 1.7B) — but that test was
    on the *compute-bound* Talker. The *bandwidth-bound* CP is untested (see 21.1).
  - Run-to-run non-determinism is **benign** (±1 LSB / −90 dB, FP rounding, not a bug).

### 21.1 Hybrid mixed-precision quant map — THE plan

**What's quantized TODAY (ground truth from the quantize fns):**

| Component | Output | INT8 today | Kept bf16/f32 |
|---|---|---|---|
| **Talker** (28 layers) | **sampled** (temp 0.5) | wq/wk/wv/wo, gate_up, down — **only if hidden≥2048** (`talker.c:188`) → **1.7B only**; **0.6B Talker stays bf16** | norms (f32), text_embedding, text_projection, codec_head, codec_embedding |
| **Code Predictor** (5 layers, hidden=1024 both) | **greedy/argmax** (temp 0) | wq/wk/wv/wo, gate_up, down **+ 15 lm_heads** — **both models** | 15 codec_embedding (lookup), norms, mtp_projection |
| **Speech decoder** (ConvNet) | deterministic | **nothing** | all f32; runs on overlapped pthread → **off critical path** |

Two structural facts drive the map:
- **CP is greedy** → far more quant-tolerant than the sampled Talker (argmax must flip to
  change output; the Talker has a sampling butterfly effect). AND it's the bottleneck
  (74–90% of frame time) AND bandwidth-bound (re-reads ~120 MB of weights 16×/frame).
  → **most aggressive quant target.**
- **Decoder is off the critical path and f32** → quantizing gives no latency win and can
  only hurt audio. **Leave it.** (This is the "not everything, only where it helps" of DS4.)

**Proposed hybrid bit assignment:**

| Comp. | Tensor | Today | **Hybrid** | Rationale |
|---|---|---|---|---|
| **CP** | FFN gate_up + down | int8 | **int4 group-wise (Q4_0, 32-blocks)** | biggest bytes + greedy-tolerant + bw-bound → int4 should finally WIN here |
| | q/k/v/o | int8 | **int4↔int8 (measure)** | smaller, Q/K→RoPE→KV slightly more sensitive |
| | 15 lm_heads | int8 | **int8 (try int4)** | logits but greedy |
| | codec_embedding ×15 | bf16 | **bf16** | sparse lookup, ~0 bandwidth |
| | norms | f32 | **f32** | 1-D, scales everything |
| **Talker** | FFN gate_up + down | int8 (1.7B) / bf16 (0.6B) | **int8** (measure 0.6B: was compute-bound → 0%) | sampled → **int8 ceiling**, no global int4 |
| | q/k/v/o | int8 (1.7B) | **int8** | feeds KV → persists across all future tokens |
| | codec_head (output) | bf16 | **bf16** | **sampled** output = most sensitive of all |
| | text_embedding | bf16 | **bf16** | huge but sparse lookup |
| **Decoder** | conv | f32 | **f32** | off critical path + audio quality |

- [x] **EXPERIMENT #1 — int4 on the CP FFN: REFUTED (2026-06-01, `feat/int4-cp-ffn`).** Wired
  Q4_0 on CP gate_up+down (forward prefers q4 > int8 > bf16), measured via `make cp-microbench`
  (0.6B, seed 42, ryan, IT, vs `--int8` baseline). int4 **LOSES on both axes**:
  - Speed: FFN gate_up 17.2→**24.8 ms/f (+40%)**, FFN down 10.2→**13.7 (+30%)**, CP TOTAL
    57→**67 ms/f (+14-18%)**. (microbench stable ~4%; single-shot wall RTF is noisy — trust the
    per-op slots, not RTF.)
  - Quality: audio rms 0.085→**0.063 (−26%)**, peak 0.72→0.53, greedy codes flipped (119→123 frames).
  - Two causes: (1) `q4_0_matvec_inner` is **1-row, not 2-row-fused** like int8 → 2× the x-loads +
    nibble-unpack overhead masks the 30MB-vs-60MB bandwidth saving; (2) per-block-32 absmax int4 is
    **too coarse for the CP** — quality loss is *independent of kernel maturity*. Even a perfect
    2-row q4_0 kernel would not fix the −26% rms. **This is NOT a DS4-style "no quality loss" win.**
  - **Decision: drop int4 on the CP.** int8 is the quality floor; the speed lever is making the
    *already-winning* int8 FASTER (SDOT native dot), not going lower-bit. → Experiment #2 below.
- [x] **EXPERIMENT #2 — SDOT native int8 dot: WON (2026-06-01, `feat/int8-sdot`).** Replaced int8
  dequant→f32→FMA with `vdotq_s32` (4 int8×int8 MACs/instr, 2-row fused) + dynamic per-vector int8
  quant of activation `x` (`quantize_act_int8`). Apple clang defines `__ARM_FEATURE_DOTPROD` by
  default on Apple Silicon (no flag; `make` CC=gcc → /usr/bin/gcc = Apple clang). Runtime opt-out
  `QWEN_NO_SDOT=1` kept as safety fallback + A/B knob. Validated across the matrix (seed 42, IT,
  cp-microbench, A/B via the env toggle):
  | Model / voice | speed | quality |
  |---|---|---|
  | 0.6B preset (CP int8) | FFN gate_up −25%, down −28%, **CP TOTAL −24%**, **RTF→0.98** | clean (listened) |
  | 0.6B Silvio `.qvoice` short | CP −30% | same duration+peak, clean |
  | 0.6B Silvio `.qvoice` long 30s | CP −19% | clean, rms +23% |
  | 1.7B preset (Talker+CP int8) | **Talker −23%, CP −21%** (−22%/frame) | clean (listened) |
  - **Key: weights stay int8** (the validated precision) — SDOT only changes *how* the dot is
    computed. The only new approximation is mild int8 activation quant. rms drifts both directions
    by utterance (sampling/trajectory variance, NOT systematic degradation). **This IS the DS4-style
    clean win** — speed with no quality loss, unlike int4 which lowered weight precision.
  - **0.6B broke the RTF 1.0 barrier** (preset short, RTF 0.98) — the Phase 18 target.
  - Activation quant is per-vector absmax; outlier worry did not materialize — listening confirmed
    fine on the greedy CP AND the sampled 1.7B Talker. Selective-SDOT (down_proj legacy) NOT needed.
- [ ] `[MED]` **Extend SDOT to CP lm_heads.** `qwen_argmax_matvec_int8` (15 lm_heads/frame, 4.3% of
  CP) still uses the f32 path — small but free. (VNNI is the x86 twin, 21.3, needs the rented box.)

> ⚠️ **SDOT is ARM-only (`vdotq_s32`, guarded by `__ARM_FEATURE_DOTPROD`).** On x86 the whole
> int8 matvec already falls to **scalar** (no AVX2 — `qwen_tts_kernels_avx.c` is empty, see 21.5),
> so x86 gets neither the SDOT win nor even the NEON int8 path; it runs scalar + single-thread.
> **TODO (x86, after ARM lands):** write the equivalent native int8 dot with **VNNI**
> (`_mm256_dpbusds_epi32` / AVX512-VNNI) PLUS the baseline AVX2 int8/bf16 matvec it builds on, and
> **re-audit the full x86 path** (today it's all scalar — measure real x86 RTF, it has never been
> benchmarked). Gated on the rented AVX512 box + Intel SDE in CI (see 21.3 / 21.5). The activation
> quant (`quantize_act_int8`) is portable C and reusable for the VNNI path.
- [ ] `[LOW]` No `silvio_17b.qvoice` to test custom voice on 1.7B (needs `qwen3-tts-1.7b-base` to
  create). 1.7B preset already validates the Talker int8+SDOT critical path.

> **Next session (2026-06-02) — validation & release gate.** Before merging `feat/int8-sdot`:
> 0. **FIRST — deep re-analysis of all docs/md/code to re-verify the real x86 state.** The
>    "x86 matvec is all scalar / AVX file empty" claim came from one Explore pass + a grep; the
>    user expected x86 support to exist and was surprised. Re-confirm THOROUGHLY whether any
>    x86/AVX2 matvec code exists before planning the VNNI work (this repo has a history of
>    assumptions believed-then-wrong — verify, don't trust the first pass).
> 1. Download `qwen3-tts-1.7b-base`, create `silvio_17b.qvoice`, test custom voice on the 1.7B.
> 2. Massive total regression: `make test-small`/`test-large`, server (bf16 + int8 + `.qvoice`),
>    voice-clone e2e, both models, SDOT on AND off (`QWEN_NO_SDOT`).
> 3. IF all green AND Phase 21 plan is considered done → merge `feat/int8-sdot` → `feat/labs`
>    (then evaluate → `main`) AND cut a release. ("vediamo" — gated on the tests.)
- [ ] `[HIGH]` **Group-wise scales for int4** (not per-row absmax like int8). Q4_0 already does
  per-32-block scales — that's the "no big quality loss" enabler. Per-row int4 = quality death.
- [ ] `[HIGH]` **Native int8 dot (ARM SDOT)** — M1 has `__ARM_FEATURE_DOTPROD`. Today int8 does
  dequant→f32→FMA (~10-14 SIMD ops / 16 weights, mostly conversion). SDOT = 4 int8×int8 MACs
  in ONE instr. For int4: unpack nibble→int8 in-register THEN sdot (cheaper than →f32).
  Requires dynamic int8 quant of the activation `x` (per-vector absmax, cheap). (folds in
  old Phase 15.)
- [ ] `[MED]` **Fuse dequant + matvec + activation** where adjacent (CP gate_up→SwiGLU).
- [ ] **int2/int1 verdict: research only.** Only the greedy CP could plausibly survive, but
  with simple absmax/group (no QAT — we're inference-only) quality dies, and ~15 MB CP at int2
  is still out of L2 → modest bandwidth gain over int4. Realistic floor = **int4-CP / int8-Talker**.

> Cache math (why low-bit only helps the CP): CP weights don't fit L2 (~12 MB M1) at any
> precision — bf16 120 MB, int8 60, int4 30, int2 15. So the win is purely bytes-from-DRAM,
> **linear in bits**, no residency threshold. The Talker is compute-bound, so fewer bytes
> don't help it (proven: int4 lost on Talker). This is why int4 belongs on the CP, not globally.

### 21.2 Cross-OS threading (Win + Linux + Mac)

Today the matvec dispatch is **Apple/GCD ONLY** (`#if defined(__APPLE__) && defined(__BLOCKS__)`
at `kernels.c:422/444/744/762/936/958` — verified 2026-06-03). Off macOS the block compiles out
→ **single-thread** decode. So every quant win currently exists **only on Apple Silicon**. The
ONLY `pthread_create` in the whole repo is `qwen_tts.c:1184` (the decoder-overlap background
thread, a single thread — NOT parallel matvec). No `#pragma omp`, no pthread pool, no Win32.
**User requirement (2026-06-03): threading must be offered on Linux/Windows too, not just Mac
ARM.** This is the single biggest cross-OS gap — even with NEON, Linux ARM is ~3–4× slower than
the bench numbers purely from running 1 core.

- [ ] `[HIGH]` **`qwen_parallel_for(n, fn, ctx)` abstraction**, one API, 3 backends, all
  **persistent pool** (workers spawned once, parked on condvar — NEVER spawn-per-matvec: the
  CP does ~80 matvecs/frame × 16 passes):
  - **macOS** → keep `dispatch_apply` (GCD) — the fast path (pthread pool was −8% vs GCD *on
    Mac*; keep it there only).
  - **Linux / WSL / POSIX** → pthread persistent pool (cond var + atomic row counter). No GCD
    here → pthread pool ≫ single-thread (Mac's −8% is irrelevant vs 1 core).
  - **Windows native** → Win32 threads + condition vars (or pthreads-w64). POSIX path is nearly
    free since the decoder already uses pthread; Windows is the delta.

### 21.3 x86 enablement (logic/ops we support there too)

**STEP 0 deep re-audit DONE (2026-06-03, verified by reading every guard + the AVX2
intrinsic tally, not a single grep).** The prior "x86 is all scalar" was *substantively*
correct but mis-framed — the truth is sharper:

**AVX2 exists, but ONLY on 5 auxiliary/elementwise ops** (the `<3%`-of-time overhead):
`qwen_rms_norm`, `qwen_rms_norm_residual`, `qwen_rms_norm_per_head`, `qwen_bf16_accum_f32`,
`qwen_bf16_to_f32_vec`. All 56 `_mm256_*` intrinsics in the repo live in these 5 (all in
`qwen_tts_kernels.c`; `qwen_tts_kernels_avx.c` is genuinely 0 lines of code).

**Everything on the HOT path falls to scalar on x86** (2-way `#ifdef __ARM_NEON … #else`
blocks, NO `#elif __AVX2__`):
- matvecs: `bf16_matvec_fused` (314), `int8_matvec_fused` (584), `q4_0_matvec_inner` (869),
  `qwen_argmax_matvec_bf16` (1624) / `_int8` — **the 90.7% of decode per the microbench**
- `int8_matvec_sdot` + `quantize_act_int8` are `#if __ARM_FEATURE_DOTPROD` → **don't exist
  at all on x86** (no scalar equiv either; callers fall back to `int8_matvec_fused` scalar)
- attention: `qwen_causal_attention` / `_windowed` / `_bf16kv`
- inline NEON-only (no AVX2) elsewhere: f32→bf16 pack + bf16→f32 + NeoX RoPE in
  `talker.c`/`code_predictor.c`/`speech_decoder.c`, `qwen_apply_rope_interleaved`,
  `qwen_swiglu`/`silu`/`add`/`mul`/`vec_scale`, `qwen_snake_activation` (decoder)
- decode-step calls the hand matvecs (`qwen_matvec_*`), NOT BLAS → scalar; only **prefill**
  uses `cblas_sgemm` (OpenBLAS) so prefill is the one fast thing on x86.

> ⚠️ **The comments in `qwen_tts_kernels_avx.c` are FALSE/stale** — they list "bf16_matvec_fused:
> AVX2", "int8_matvec_fused: AVX2", "qwen_causal_attention: AVX2", "qwen_snake_activation: AVX2"
> as "Active AVX optimizations". NONE of those have AVX2. This stale doc is very likely what made
> us believe x86 was covered. **Fix the comments (and `_neon.c`'s, which over-claim too).**

**Goal (user, 2026-06-03): every NEON-accelerated op must have at least an AVX2 twin + a
scalar fallback ALWAYS; AVX512/VNNI optional on top.** Plan:

- [ ] `[HIGH]` **AVX2 twin for EVERY hot op** (not just matvecs): the 4 matvecs + attention +
  RoPE + bf16 pack + swiglu/add/mul/scale + snake. Keep scalar fallback in all. This is the
  bulk of the work — x86 has zero hot-path SIMD today.
- [ ] `[HIGH]` **Runtime cpuid dispatch + drop `-march=native` for release** (`Makefile:5` →
  SIGILL on older CPUs). Baseline `-mavx2 -mfma`; function-multiversioning or a dispatch table
  so one binary runs SSE→AVX2→AVX512(-VNNI/-BF16) by detected ISA.
  - **PRINCIPLE (user, 2026-06-03): at runtime ALWAYS pick the BEST extension the CPU supports,
    else step down — AVX512 → AVX2 → SSE → scalar (and on ARM: i8mm/bf16 → SDOT → NEON → scalar).
    Never compile-time-lock to the build machine's ISA.**
  - **Real-world evidence:** a user on a brand-new Ryzen (should have AVX-512) reported SLOW perf.
    Cause = exactly this gap — we have ZERO AVX2/AVX512 on the hot path AND no runtime dispatch, so
    his AVX-512 silicon ran our scalar single-thread decode. This is the bug to kill.
- [ ] `[MED]` **VNNI** (`_mm256_dpbusds_epi32` AVX-VNNI / AVX512-VNNI) = x86 SDOT twin for int8,
  on the rented AVX512 box. Prereq: AVX2 baseline + dispatch above. `quantize_act_int8` is
  portable C and reused as-is. Verify ISA with Intel SDE in CI; tune perf only on real HW.
- [ ] `[LOW]` **AVX512-BF16** (`_mm512_dpbf16_ps`) bf16 dot twin, same box.
- **Order**: finish ARM (incl. the NEON headroom in 21.3b, dev HW measurable now); x86 after, on
  a rented AVX512 server (can't tune blind on M1).

### 21.3b ARM NEON is NOT at peak either (the SDOT lesson, generalized)

Just like SDOT was *missing on NEON* until 2026-06-01, the NEON matvecs still leave perf on the
table on **post-M1 ARM** (M2/M3/M4, Graviton3/4) — all behind `#if __ARM_FEATURE_*` with a
NEON→scalar fallback, so M1 and x86 are unaffected:

- [ ] `[MED]` **bf16 native dot** — `bf16_matvec_fused` does dequant(`vshll`)→FMA. ARMv8.6
  (`__ARM_FEATURE_BF16`: M2+) has `vbfdot`/`vbfmmla` for native bf16 MACs → skips the widen.
- [ ] `[MED]` **int8 i8mm** — SDOT (`vdotq_s32`) is 1×vector; `smmla` (`__ARM_FEATURE_MATMUL_INT8`:
  M2+, Graviton3+) does a 2×2 int8 matmul per instr → ~2× on the int8 matvecs. The 2-row-fused
  layout already half-fits smmla's shape.
- [ ] `[LOW]` **SVE/SVE2** — Graviton3/4 + ARM servers: vector-length-agnostic loops. Research only.

### 21.4 INT8 productization

- [ ] `[MED]` Evaluate making `--int8` the **default** — 0.6B validated good on Apple; **must
  measure x86/Linux first** (scalar+single-thread there → int8 could regress vs bf16). Don't
  flip blind.
- [ ] `[MED]` **On-disk int8 converter**: ship pre-quantized model files (~half size), no runtime
  quant, faster load. Add a load path for pre-quantized weights. Keep the download-originals path.
- [ ] `[MED]` **int8 deltas in `.qvoice`/WDELTA** (currently int16 → ~half the file).
- [ ] `[LOW]` `download_model.sh --int8`: pull pre-quantized models from our HF repo.
- **LEGAL ✅**: Qwen3-TTS-12Hz CustomVoice is **Apache-2.0** → quantize + redistribute on our HF
  account is allowed (keep LICENSE/NOTICE, attribute Qwen, state changes; `Naumius/` precedent).

### 21.5 Cross-CPU coverage — VERIFIED reality (2026-06-03 deep re-audit)

| Platform | Hot-path SIMD (matvec+attn = 90%) | Aux SIMD (rms/bf16, <3%) | Decode threading | Reality |
|---|---|---|---|---|
| **macOS ARM (M1)** | NEON ✅ (+SDOT int8) | NEON ✅ | GCD 4-thread ✅ | the only truly optimized target (= all benchmarks) |
| **macOS/Linux ARM (M2+/Graviton)** | NEON ✅ but **bf16/i8mm headroom** (21.3b) | NEON ✅ | GCD ✅ Mac / **single ❌** Linux | works, leaves ~2× on the table on the matvecs |
| **Linux ARM (aarch64, M1-class)** | NEON ✅ | NEON ✅ | **single-thread ❌** | correct but ~3–4× slower than bench (1 core) |
| **Linux/WSL/Win x86-64** | **scalar ❌** (no AVX2 on any matvec/attn) | **AVX2 ✅** (only these) | **single-thread ❌** | decode catastrophic; only prefill (`cblas_sgemm`) is fast |

**Coverage matrix we OWE users** (goal: NEON-equivalent everywhere SIMD exists, scalar always):
| Op family | NEON | AVX2 | AVX512/VNNI | scalar | Gap |
|---|---|---|---|---|---|
| matvec bf16/int8/q4_0 + argmax | ✅ | ❌ | ❌ | ✅ | **AVX2+VNNI (21.3), bf16/i8mm NEON (21.3b)** |
| attention (3 variants) | ✅ | ❌ | ❌ | ✅ | **AVX2 (21.3)** |
| RoPE / bf16 pack / swiglu / add·mul·scale / snake | ✅ | ❌ | ❌ | ✅ | **AVX2 (21.3)** |
| rms_norm ×3 / bf16_accum / bf16_to_f32 | ✅ | ✅ | ❌ | ✅ | AVX512 optional |

Fixing this = 21.2 (threading) + 21.3 (AVX2 on ALL hot ops + VNNI) + 21.3b (post-M1 NEON).
Until then, all RTF numbers are **Apple-M1-only**.

---

## OPEN FUTURE TASKS (compact — nothing dropped)

- [ ] `[MED]` **Server warm-request reproducibility bug (found 2026-06-03).** `make test-serve-bench`
  fails its bit-identical assertion: two IDENTICAL consecutive requests (seed 42, same text) produce
  DIFFERENT output — run1 291884 B, run2 311084 B (~400 ms longer), even at `-j1 --temperature 0`.
  PROVEN **pre-existing & NOT SDOT** (fails identically with `QWEN_NO_SDOT=1`; engine itself is
  bit-deterministic — CLI `-j1 temp0` is bit-identical, and the server's COLD run1 == CLI output).
  Cause = server **delta-prefill / KV-reuse across requests**: the warm 2nd request doesn't get a
  clean state, so it diverges. Contradicts the "delta prefill = bit-identical" belief. Fix: reset KV
  state between unrelated requests OR only reuse on a verified-matching prefix. The `test-serve-bench`
  md5 assertion is also too strict (should compare cold-vs-cold, or mel-corr, not warm-vs-cold md5).
- **CP sliding window attention** (old 18.3): config has `sliding_window=72`; verify CP attention
  caps at it. `[MED]`, only matters for 200+ frame sequences.
- **Long-form / audiobook mode** (old Phase 19): chapter/batch mode, progress indicator + ETA/RTF,
  long-run telemetry (RSS, RTF drift, thermal), resumability, `make test-longform`. Plus a
  lightweight bracket markup (`[pause:500]`, `[emph]…[/emph]`, `[voice:…]`, `[instruct:…]`) parsed
  by a tiny hand-written tokenizer in a new `qwen_tts_script.c` (no XML, zero deps). `[MED/LOW]`
- **AVX512 hand-kernels** (old 20.5): defer. BLAS already dispatches AVX512 at runtime; our hand
  kernels are memory-bound. Only add `#ifdef __AVX512F__` paths (fallback + SDE-verified in CI) if
  profiling on real HW shows the aux kernels hot. `[LOW]`
- **Metal GPU / MLX** (old Phase 16): was 1.3× slower on M1 (unified-memory bandwidth ceiling).
  Revisit only on M3/M4 (higher bandwidth + Metal 4 tensor support). `[LOW]`
- **Windows native** (old Phase 15): mmap→MapViewOfFile, threads (covered by 21.2). WSL2 works
  today and is the recommended path. Only if real demand. `[LOW]`
- **Server per-request voice switching**: accept voice path in JSON (WDELTA too heavy for hot-swap
  today; needs a lighter per-request mechanism). `[LOW]`
- **KV cache prefix caching eval** (old Phase 12): is a ~50MB KV prefix dump worth it vs WDELTA?
  Likely no (WDELTA is bit-identical + portable), but un-evaluated. `[LOW]`
- **CUDA/HIP backend stubs**, **top-p partial sort with early exit** (only when top_p<1.0). `[LOW]`

---

## DONE — history (compacted)

### Core engine & features
Full pipeline (Talker → Code Predictor → Speech Decoder → WAV), both model sizes (auto-detected),
HF safetensors loader (mmap), 9 preset speakers / 10 languages, instruct/style control (1.7B),
streaming (`--stream`/`--stdout`/callback), HTTP server (OpenAI-compatible), voice cloning (Base:
ICL + x-vector), VoiceDesign (1.7B), `--max-duration`/`--seed`/EOS boosting, `download_model.sh`,
Makefile test suite, WSL2 build docs. CI/CD: GitHub Actions build matrix (Linux x86/ARM, macOS
ARM/x86), CodeQL, clang-tidy, ASan/UBSan, release artifacts on tag.

### Performance optimizations (landed)
BLAS (Accelerate/OpenBLAS), NEON SIMD kernels (rms_norm, attention, RoPE, bf16↔f32, add/mul/scale),
cache-line-aligned buffers (+24%), LRU text-embedding cache, **decoder thread overlap** (pipeline
parallelism), multi-row bf16 matvec (2-row fused), unified QKV dispatch, fused gate+up, top-k
quickselect (4× sampling), batch vvexpf SwiGLU, delta prefill / KV reuse (server, ~50% prefill),
persistent prefill buffers, BF16 KV cache, **fused residual+RMSNorm** (−21% short), software
prefetch (~1-2%). INT8 + streaming TTFA: see 21.0.

### CP micro-bench (June 2026, `make cp-microbench`, 0.6B, 114 frames)
CP = **90.7% matvec/GEMV** (FFN gate_up 34% + down 19% = 53%; QKV 24.5% + O 13.3% = 38%; lm_head
4.3%). All "overhead" (attention 0.7%, RoPE/norm/KV-store/dispatch) = **<3% combined**. → CP is
matvec/**bandwidth**-bound, NOT overhead-bound. `matvec_bf16` is hand NEON, not BLAS. This refuted
the old "CP is overhead-bound" hypothesis (the 18.1 micro-opts targeting <3% were dropped).
Baseline: CP 86.6 ms/f (74%), Talker 30.6 (26%), sampling 0.35 (negligible), prefill 1.65s (TTFA),
decoder overlapped. RTF 1.76.

### Experiments that DID NOT work (don't re-litigate)
- **Metal GPU**: 1.3× slower on M1 (unified-memory bandwidth ceiling). Removed.
- **pthread thread pool vs GCD on macOS**: −8% (GCD is kernel-optimized). → keep GCD on Mac, but
  pthread is the right call OFF Mac (see 21.2).
- **4-row fused matvec**: −7% vs 2-row (register spill). Threading threshold 2048: −3%.
- **INT4 Q4_0 on the Talker**: slower than int8 (compute-bound; nibble unpack > bandwidth saved).
  Kept opt-in. **Untested on the CP** → see 21.1.
- **INT8 on the 0.6B Talker**: 0% (hidden=1024 too small / compute-bound). The 0.6B win is CP-only.
- **Speculative CP decoding**: abandoned (codebook feedback loop is structurally unsafe → also why
  the 16 CP passes can't be batched).
- Batch text-embedding sgemm (0.13%), softmax SIMD (post-quickselect), depthwise-conv/LayerNorm
  SIMD (decoder overlapped), NEON SiLU (<1%): all skipped as non-bottlenecks.
- **CP↔Talker pipeline overlap**: infeasible (sequential dependency).

### Non-determinism — investigated, BENIGN
Output not always bit-reproducible run-to-run, but diff = **±1 LSB of 16-bit PCM (~3e-5, −90 dB,
corr 1.0000000)** = FP rounding, not audible, not a bug. Two causes: (1) temp>0 sampling + threaded-
matvec FP noise → occasional token flip (vanishes at `--temperature 0`); (2) overlapped decoder
variable chunk timing → ±1 LSB (load-dependent; quiet machine → bit-identical). NOT uninitialized
memory, NOT seam artifacts (both disproven). Decision: chunk-invariant decoder NOT pursued (would
add overhead to fix a −90 dB diff). For bit-stable A/B: `-j1 --temperature 0`, else compare by
RTF + mel-corr, not md5.

### Cross-model voice / `.qvoice` saga (Phases 12–13, March 2026)
Goal: reusable custom voices + cross-model injection (clone on Base → use on CustomVoice w/ instruct).
Key discovery: Base and CustomVoice transformer weights are **nearly identical** (cosine
0.9998–1.0000); the only real differences are the codec_embedding speaker presets + Base's 76
ECAPA-TDNN tensors. Cross-model divergence ≈ same-model seed variance (butterfly effect from
`tts_pad_embed` micro-diff + autoregressive sampling), NOT a fixable bug. Progression of formats:
`.bin` x-vector (~60-70% fidelity) → full KV dump (~90%, but text-locked) → **WDELTA `.qvoice`**
(the shipping HQ format): int16 weight deltas + LZ4, **bit-identical** cross-model, works with
`--instruct`, +7% load overhead, target CV size validated in header. `--save-voice` needs Base+CV
present; usage needs only CV + the `.qvoice`. Server preload supported. (WFULL 1.7GB proved
bit-identical is possible; WDELTA compressed it to ~785MB LZ4.) `.qvkv` KV-prefix approach abandoned
(WDELTA superior in every dimension). Long tail of fidelity experiments (lower temp, longer ref,
greedy warmup, partial-layer replacement) all WORSE — 30s ref is the sweet spot, it's all-or-nothing.

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS) ·
  [Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- HuggingFace: [0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) ·
  [1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- Community: [FastAPI server](https://github.com/ValyrianTech/Qwen3-TTS_server) ·
  [OpenAI API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi) ·
  [streaming](https://github.com/rekuenkdr/Qwen3-TTS-streaming) ·
  [faster-qwen3-tts (Andres Marafioti)](https://github.com/andimarafioti/faster-qwen3-tts)
