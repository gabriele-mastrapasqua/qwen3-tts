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
- [x] `[REFUTED 2026-06-04]` **SDOT on CP lm_heads — DON'T.** Tried routing `qwen_argmax_matvec_int8`
  through the SDOT path (quantize x to int8, vdotq_s32). It is NOT free: quantizing the lm_head
  activation flips enough near-tie argmaxes that — via the CP→Talker code feedback — the int8
  trajectory FORKS (int8 golden mel-corr 1.0 → **0.51**, duration +11%). The f32-precision activation
  on this one final matvec is what STABILISES the int8 trajectory. ~0.9% overall speed (4.3%-of-CP ×
  ~20%) is not worth changing the validated int8 output. Kept on the f32 path; documented in-kernel.
- [x] `[NO-WIN 2026-06-04]` **q4_0 2-row fuse — reverted.** 2-row-fused the q4 kernel to share x-vector
  loads across two output rows (bit-identical, verified). Measured on M1 NEON (-j1, 0.6B int4): CP
  ~167.7 vs ~164.7 ms/f = **no gain, within noise / slightly worse**. The q4 nibble DECODE (unpack +
  cvt-f32 + scale) dominates; x-loads were never the bottleneck on NEON (the workflow's "+40% x-loads"
  applied to an older kernel state). Added register pressure (8 x-vectors + 2 accumulator sets) didn't
  help. Reverted per "no complexity for tiny wins." Re-try only if x86/AVX2 measurement shows x-loads
  matter there (the AVX2 twin would need the rented box). The real q4 lever, if any, is a DECODE
  rework (e.g. int8-x + SDOT on decoded nibbles), not x-load sharing.

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

- [x] `[HIGH]` **`qwen_parallel(nt, fn, ctx)` abstraction** — DONE 2026-06-03 (branch
  `feat/avx2-xos-threading`, `qwen_tts_thread.{c,h}`). One API, 3 backends, all **persistent
  pool** (workers spawned once via `qwen_threadpool_start`, parked on condvar, chunks claimed
  by an atomic counter; main participates). The 6 GCD-only `dispatch_apply` sites in
  `qwen_tts_kernels.c` now route through it:
  - **macOS** → `dispatch_apply` (GCD), the fast path — unchanged.
  - **Linux / WSL / POSIX** → pthread persistent pool. Was single-thread before.
  - **Windows native** → Win32 threads + condition vars (structural; untested — no Win box).
  - **Validated on M1**: GCD golden mel_corr=1.0; forced-pthread build (`-DQWEN_FORCE_PTHREAD`)
    mel_corr=1.0 & pool-count-invariant; **full x86_64 scalar+pthread binary under Rosetta →
    mel_corr=1.0 at 249% CPU** (real cross-ISA + pthread-pool end-to-end proof). `--caps` now
    reports the active pool (GCD/pthread/Win32), never SINGLE-THREAD.

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

- [x] `[HIGH]` **AVX2 twin for every hot op** — DONE 2026-06-03 (`feat/avx2-xos-threading`).
  All 4 matvecs (bf16/int8/q4_0 + argmax bf16/int8) + all 3 attention variants (score dot +
  online-softmax accumulators + bf16-KV) now have a `#elif defined(__AVX2__)` branch, 2-row
  fused / FMA, scalar fallback preserved. `qwen_quantize_bf16_to_int8` (load-time) too. Simple
  elementwise (silu/add/mul/scale, swiglu) auto-vectorize under `-ffast-math` — left as-is.
  **Still scalar on x86 (follow-up, correctness-safe):** the NEON-only inline f32<->bf16 pack +
  NeoX RoPE in talker/code_predictor/speech_decoder, and snake. **Validated:** x86+AVX2
  cross-compiles clean (`clang -target x86_64 -mavx2 -mfma`, `-Wall -Wextra`); **runtime AVX2
  correctness still owed — needs the Ryzen box** (Rosetta has no AVX2, so it can't run here).
- [~] `[HIGH]` **Drop `-march=native` off-Mac + runtime ISA guard** — DONE (the SIGILL bug).
  Makefile now: Linux x86 default `-mavx2 -mfma` (portable Haswell+), `SIMD=scalar` for
  pre-AVX2, `SIMD=avx512` opt-in; macOS/ARM keep `-march=native`. `qwen_check_runtime_isa()`
  aborts with a clear message if an `-mavx2` binary runs on a non-AVX2 CPU (verified under
  Rosetta: FATAL + exit, **no SIGILL**); `--caps` prints `runtime cpu:` + a warning. **Still
  open:** true per-CPU function-multiversioning (one fat binary auto-stepping
  SSE→AVX2→AVX512) — today it's one ISA per build. AVX-VNNI/AVX512-BF16 below.
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

## 21.6 VERIFIED full-codebase audit (2026-06-03) — master fix backlog

5 parallel audit agents + my own code verification (the lesson: don't trust agent claims either —
one falsely said "int4 is loaded but never used"; DEBUNKED, `talker.c:397-450` do call
`qwen_matvec_q4_0*`). Findings below are **verified**; "needs-verify" tagged where not yet confirmed.

**CORRECTNESS (fix first — testable on M1 now):**
- [x] **Server cross-request reproducibility — FIXED 2026-06-03 (commit cbfa979).** Root cause was NOT
  "kv_len not reset" but: on a FULL prefix match (`delta_start==prefill_len`, identical consecutive
  request) the prefill block at `qwen_tts.c:1108` was skipped entirely, leaving `ctx->dec_x` (read to
  seed the first generated frame) STALE from the previous request's last token step. Fix: on full
  match force `delta_start=0` (full fresh prefill = bit-identical to cold). Partial matches (real
  server case: shared prefix, different text) untouched → delta-prefill optimization preserved.
  Verified: 3 identical reqs now bit-identical AND == CLI (327ec448); `test-serve-bench` PASSES (was
  FAIL); test-small 5/5. Remaining server hardening (below) still open.
- [x] **Server thread-safety — single-threaded by design; mutex added as foundation (2026-06-04,
  commit a3819a2).** The server is genuinely single-threaded: one `while(server_running)` loop,
  `accept()` → `handle_*()` inline → close, NO `pthread_create`/fork. Requests are serialized; a 2nd
  concurrent curl waits in the listen backlog → no live race. Added `g_synth_lock` around the 3
  synthesis handlers anyway: UNCONTENDED today, but the correct foundation for the future
  concurrent-serving throughput feature (synthesis mutates the shared ctx).
- [x] **Server input validation — HARDENED (2026-06-04, commit a3819a2).** Added: text length cap
  (`MAX_TTS_TEXT` 8192 chars → clean 400, finer than the existing 1 MB raw-body cap) + empty-text 400;
  **sampling-param clamping** (temperature [0,**2.0**], top_k [0,vocab], top_p [0,1], rep_penalty
  [0.5,2]). The clamp is NOT cosmetic: a degenerate `temperature:99`+`top_k:0`+`top_p:1` makes
  sampling so flat the model never emits EOS and runs to `max_frames` (caught a 43-min orphaned-server
  runaway during testing). Invalid speaker/language still safely ignored; `max_frames` is the hard cap.
  ⚠ TEST-HARNESS LESSON: always `timeout` server curls + `pkill -f qwen_tts.*--serve` by name (never
  rely on `$!`/`wait` — a blocked curl hangs the script and orphans the server).
- [x] **Voice-clone 24kHz ref audio — RESOLVED as documented-by-design 2026-06-03 (commit 33c11a0).**
  Decision (user): do NOT bundle a resampler (ffmpeg does it better, keeps zero-dep; mel features need
  24kHz). The requirement is now documented in docs/voice-cloning.md + `--help` + a clear runtime error
  (`ffmpeg -i in -ar 24000 -ac 1 out.wav`). The misleading "TODO: resample" comment was rewritten.
- [x] **Robustness: tokenizer `fread()` — FIXED 2026-06-03 (commit 33c11a0).** Both vocab.json +
  merges.txt loads now check the read length (short read → error + free, was silently ignored).
  `(size_t)rows*cols` overflow: low risk, cast present; left as-is (would need absurd config dims).

**TEST & VALIDATION BACKLOG (added 2026-06-04 — quant × voice × delivery cross-product + concurrency):**
- [ ] **Re-run the FULL test matrix with `.qvoice` AFTER clone, crossed with quants.** The 0.6B-Talker
  int8 change (commit 12b73d7) + WDELTA re-quant path means voice-clone × {bf16, int8, int4} must all be
  re-validated: create a `.qvoice`, then load it under each quant and check EOS/coherence/RTF + golden-
  style mel-corr. Today's goldens cover preset voices only — NO `.qvoice`×quant golden exists.
- [ ] **DESIGN DECISION: should `.qvoice` store quantized weights?** Option A (today): `.qvoice` is bf16
  WDELTA, quantized at load if `--int8/--int4` (re-quant after WDELTA override). Option B: at voice-CREATE
  time let the user choose to quantize, SAVE the quantized weights in the `.qvoice`, and load them
  pre-quantized (faster load, smaller file, fixed precision per voice). Trade-off: B locks the precision
  into the file (can't switch), A stays flexible. Decide after the int8-default question below. If int8
  becomes the default, B (save-quantized) gets more attractive.
- [ ] **Re-run ALL server RTF benchmarks** (cold/warm, bf16/int8/int4, `.qvoice`) now that the 0.6B Talker
  is int8 under `--int8` — the README server RTF table (1.33/1.34) predates this and is stale.
- [ ] **NEW concurrency tests (for the batching work):** (a) parallel-request test — N concurrent curls,
  assert all return valid distinct WAVs (today serialized by `g_synth_lock`; this test must stay green
  when concurrency is added); (b) race detector — build a `make test-serve-race` that runs the server
  under TSan (`-fsanitize=thread`) + concurrent load to prove `g_synth_lock` actually guards the shared
  ctx (and catch any unlocked shared state when batching lands); (c) streaming RTF test + verify streaming
  works under int8/int4 (TTFA + RTF per quant). All must use the timeout+pkill harness (runaway lesson).

**PERFORMANCE / PORTABILITY (the big gap — see 21.2/21.3/21.3b):**
- [ ] All 8 hot matvec/attention kernels scalar on x86; matvec threading GCD/Apple-only; SwiGLU exp
  uses Accelerate `vvexpf` on macOS, scalar `expf` elsewhere (`kernels.c:1433`); post-M1 NEON headroom.
- [ ] `[HIGH]` **Makefile `-march=native`** (line 5) → SIGILL on older CPUs, no release/portable
  target, no runtime cpuid dispatch. **Principle: pick best ISA at runtime, else step down.**
- [ ] **CI is build-only off-Mac** — `.github/workflows` build Linux x86/ARM + macOS-ARM but run only
  `./qwen_tts --help`; **no inference is ever executed off Apple Silicon, and x86 has NEVER been
  benchmarked.** No macOS-x86, no Windows. Add a real inference smoke (small model) + SDE for ISA.
- [ ] **Windows native won't compile** (`mmap`/`pthread`/`gettimeofday`, no Win32 fallback) — WSL2 only.

> ✅ **x86 test box NOW available (2026-06-03):** user's **Ryzen 7 6800H** mini PC (Zen 3+, **AVX2+FMA
> yes, AVX-512 no**) on LAN `192.168.1.93` (RDP; see memory `reference_x86_test_box`). Build under
> **WSL2** (our Linux x86 path), run a small model, confirm AVX2 is active + get real x86 perf.
> Workflow: Claude gives build/debug commands → user runs over RDP → pastes logs → verify together.
> ⟹ This validates the **AVX2 baseline matvec + x86 threading** (the 90% gap). **VNNI/AVX-512** still
> needs a Zen4+/Intel box. Approach (user): **write + correctness-verify AVX2/threading now** (scalar-
> equivalence + SDE-in-CI), then **measure on the Ryzen box** — no longer fully blind. (This is exactly
> the testing that never happened before → why x86 silently rotted.)

**DEBUNKED agent claims (do NOT propagate):** "int4 loaded but never used" (FALSE — int4 wired in
talker.c forward). "x86 FTZ incomplete" and ".qvoice v1 enc_dim hardcoded" — `needs-verify` before acting.

### 21.7 Test coverage — hardened 2026-06-03 (safety net BEFORE the AVX2 work)

The old suite only proved the pipeline RUNS (`validate_wav` = non-empty WAV + ≥1 frame + no MISSING
weights) — it never checked the audio was CORRECT. A numerically-broken kernel that still emits audio
PASSED. Closed the worst gaps:
- [x] **`test-golden`** (commit d987c4b/d987... + gitignore fix) — regen deterministically (`-j1
  temp0 seed42`) and compare to committed `tests/golden/*.wav` via **mel-spectrogram correlation
  (≥0.99) + duration (≤5%)** (`tests/compare_audio.py`, librosa). Covers 0.6B en/it/int8 + 1.7B en.
  Wired into `test-all`. `make golden-update` regenerates after an intended change. **mel-corr (not
  md5)**: md5 flakes even at `-j1 temp0` (±1 LSB decoder noise, verified) AND mel-corr is the correct
  cross-ISA check for AVX2/x86 (won't be bit-identical, must stay ~0.99+). This is THE safety net for
  the kernel work — a wrong AVX2 kernel now fails here instead of silently shipping.
- [x] **`test-serve-repro`** — 3 identical requests bit-identical (catches cross-request state leaks).
- [x] **`test-voice-design`** now SKIPs cleanly (was failing on absent model — double bug: dir-only
  check + per-line `exit 0` that didn't stop the recipe).
- [x] **`--caps` + `test-caps`** — the binary reports its ACTUAL compiled SIMD/threading (NEON/AVX2/
  scalar, SDOT, GCD/single, BLAS), and the test asserts arch↔caps consistency. **This is the
  "would-have-caught-we-thought-AVX-existed" guard**: on x86 it reports `matvec+attn: SCALAR` +
  `SINGLE-THREAD`, impossible to hide behind docs. When AVX2 lands, flip the x86 assertion.
- [x] **`test-errors`** — bad invocations (no --text/--serve, nonexistent model, missing .qvoice)
  must fail cleanly (non-zero + clear message). Fast, no model.
- [x] **Variance characterized** — quiet machine + fixed seed → mel_corr 1.00000 across det/-j1-temp0
  AND default-4thread-temp0.5 (18 runs). So Qwen3-TTS is NOT audibly non-deterministic with a fixed
  seed on a quiet box ("poco" = ≤±1 LSB). Golden threshold set to 0.98 (margin), cross-ISA 0.95.
- [ ] `[LOW]` **Remaining smaller gaps:** SDOT on/off A/B not in suite, `.qvoice` load (local-only,
  voices/ gitignored → skip-if-absent), int4 not in golden, truncated-file fread path (no test), **no
  x86/Linux inference in CI** (build-only — add once the Ryzen/WSL2 flow is set up).
- [ ] `[LOW]` **`test-clone` has the same latent per-line `exit 0` skip bug** as voice-design had — not
  currently broken (base-small model present) but would FAIL instead of SKIP if absent. Same one-shell fix.
- [x] **RESOLVED — there is NO engine non-determinism bug (2026-06-03).** On a clean, confirmed
  `-O3` binary (0 ASan symbols) on a non-saturated machine, `-j1 --temperature 0` is FULLY
  deterministic: 5/5 runs byte-identical TOKENS (82505ec4) AND audio (327ec448 = golden). The entire
  "non-determinism" hunt was a MEASUREMENT ARTIFACT of my own making: (1) **binary mixing** — after
  `make debug` the binary was ASan **-O0** (different FP rounding/order than `-O3 -ffast-math` → a
  legitimately different trajectory), and `make blas` did NOT rebuild it (timestamps), so I compared
  -O0 vs -O3 outputs and called the difference "drift"; (2) **my own runaway busy-loops + leftover
  processes** saturating the 4-core box during other measurements. Lesson (again): verify the binary
  + a quiet machine before trusting an A/B; `make debug` then `make blas` needs a `make clean` between
  (timestamp trap). The QWEN_NO_OVERLAP knob (added during the hunt) is harmless + kept as a
  decode-overlap diagnostic. **NOTE for product:** at DEFAULT (multi-thread, temp>0) output is
  legitimately "similar not identical" (FP reduction-order across threads + sampling) — that's why the
  golden test uses mel-corr, and that benign variation is expected, not a bug.
- ~~`-j1 --temperature 0` intermittently non-deterministic / likely UB~~ — **WRONG, fully retracted**
  (see the RESOLVED entry above). It was binary-mixing (ASan-O0 vs -O3) + my runaway processes, not an
  engine bug. Kept here only as a record of the false trail so it isn't re-opened.

---

## OPEN FUTURE TASKS (compact — nothing dropped)

- [ ] `[LOW, AFTER fixes]` **Correct the BLOG docs' false x86/AVX claims.** `blog/optimization-notes.md`
  (and related) repeat "NEON on ARM, AVX on x86" for matvecs/attention — FALSE (only rms_norm +
  bf16_accum have AVX2). Authoritative docs (README/CLAUDE/docs/*) fixed 2026-06-03 (commit 7aa0f9b);
  the blog narrative is deferred until AFTER the real fixes + fresh measurements land, so it's
  rewritten once against true numbers rather than patched twice.
- [x] **Server warm-request reproducibility bug — FIXED 2026-06-03 (commit cbfa979).** See 21.6 for
  the verified root cause (stale `ctx->dec_x` on full prefix match) + fix (full match → `delta_start=0`).
  `test-serve-bench` now PASSES. (The md5-bit-identical assertion turned out fine once the bug was
  fixed — it was the bug, not the test, that was wrong.)
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

## Future research (discovered 2026-06-04, to re-analyze)

### A. Prosody/emotion control by perturbing the Code Predictor (instruct is too weak)
Serendipity: quantizing the CP `down` projection to 2-bit (`QWEN_CP_Q2_FFN=down --int4`)
makes the voice **intelligible but aggressive/rough** ("death metal"). Perturbation magnitude
maps to roughness: int8 = identical to bf16, int4 = slightly aggressive, q2 = strong. Mechanism:
RVQ split — Talker codebook 0 = words (intact), CP codebooks 1-15 = fine texture/prosody;
perturbing the CP roughens texture without losing intelligibility.
- **Why it matters:** `--instruct` barely changes delivery (known Qwen3-TTS limitation, even
  full-size on GPU; community-reported). The acoustic emotion signal is weak in the output.
- **Principled lever = activation steering / control vectors:** diff CP activations
  (neutral − instruct-angry) → an "emotion direction" → inject it amplified (×N) at inference →
  controllable, audible emotion (amplify the weak instruct signal instead of breaking weights
  at random). The quant-roughness is the crude cousin of this.
- **Experiments:** (1) tunable `--roughness` = blend bf16↔q2 output on CP `down`
  (down_out = (1-r)*bf16 + r*q2); (2) control-vector emotion injection on the CP; per-layer and
  per-speaker/language sensitivity unknown (q2 effect confirmed only on 0.6B/ryan/EN/seed42).

### B. Weight-stationary batching = the throughput lever for the SERVER
Single-stream (one audio) cannot reorder to keep weights cache-hot: the CP's 16 steps (and the
Talker's tokens) are a hard autoregressive chain (step g input = step g-1 output), so the weight
read order `L1..L5 x16` can't be reordered ("weights-outer, steps-inner" needs all step inputs at
once — they don't exist yet). BUT across **independent concurrent requests** the reorder is valid:
batch N requests -> each weight read is reused across all N (weight-stationary across the batch) ->
~Nx throughput. This is standard LLM continuous-batching. **Not a single-file latency win; a big
multi-request server throughput win.** Re-analyze when optimizing the HTTP server for concurrency.

### C. Break the single-stream autoregressive wall — speculative decode + contextual sparsity
The status-quo breakers used by DeepSeek/Qwen3.6 MTP etc., applied to the CP. Both could win
SINGLE-STREAM (unlike B which needs concurrent requests). Both gated on an empirical measurement.

**C1. Self-speculative decoding on the CP.** A cheap DRAFT predicts all 15 codebooks fast (e.g.
the CP run at q2 — the "death metal" build — or a tiny head); the real CP (int4/bf16) VERIFIES all
15 in ONE batched pass (parallel over positions, like prefill — machinery already exists). Weight
read amortized over 15 positions instead of 15 sequential reads = the cache/bandwidth win, on a
single stream. Speedup = f(acceptance rate). Greedy needs exact argmax match; if the q2 draft
diverges too much (we saw it does for audio quality) acceptance may be low. (NB: the CP is named
"MTP" in-code but that's the residual predictor, NOT a spec-decode draft head — don't conflate.)

**C2. Contextual sparsity (Deja Vu / PowerInfer style).** If only a fraction of the 3072 FFN
intermediate neurons meaningfully fire per token, compute only those rows → fewer weight bytes
read → bandwidth win, single-stream. Needs a cheap predictor of which neurons fire.

**FIRST STEP = MEASURE — ✅ DONE 2026-06-04 (`make quant-ladder` + `tests/quant_ladder.py`).**

**⚠ Architectural finding that reshaped the measurement:** the CP output (codebooks 1-15)
FEEDS BACK into the Talker's next-step input embedding (`qwen_tts.c:1300-1310`:
`step_embed = codec_embed(code0) + Σ cp_codec_embed(codes 1-15) + tts_pad`). So changing CP
precision forks the ENTIRE autoregressive trajectory — different `code0`, different LENGTH
(free-running q2 ran 190 frames vs bf16's 128). A naive free-running precision sweep therefore
measures NOTHING (≈1-8% agreement = pure trajectory decorrelation, near random). **This feedback
coupling is also the mechanistic reason int4/q2 audibly change prosody AND duration, not just
texture.** → measurement must TEACHER-FORCE: lay bf16 "rails" (the 16-codes/frame stream), then
replay them (`QWEN_TF_CODES`) at each CP precision (`QWEN_CP_PREC`, Talker held bf16) so every
precision sees bit-identical per-step inputs. Implemented; bf16-TF == rails = 100% (harness valid).

**Quant-ladder results (0.6B, 128 frames, teacher-forced, argmax agreement vs bf16):**

| codebook | int8 | int4 | q2(down) |
|---|---|---|---|
| c1 (coarsest residual) | 90% | 77% | 33% |
| c5  | 80% | 57% | 12% |
| c10 | 77% | 45% | 5% |
| c15 (finest) | 73% | 23% | 8% |
| **overall** | **78%** | **46%** | **9%** |

- **Drift GROWS with codebook index** — the late/fine RVQ residuals (c11-c15) are the most
  quant-sensitive. int4 holds the early residuals (c1-c5: 57-77%) but collapses on late ones
  (c12-c15: 23-27%). int8 degrades gently and uniformly (≈73-90%). q2 destroys everything past c5.
- **int4-vs-int8 = 44% overall, worst at c11-c15 (20-35%)** → int8 IS the artifact-free gold;
  int4's "slight aggression" lives in the late/fine codebooks where it diverges most from int8.
- **FFN activation sparsity (post-SwiGLU |x|<eps): 0.28% @1e-4, 2.25% @1e-3, 14% @1e-2, 59% @1e-1.**
  The CP FFN is essentially DENSE — NOT the 80-90% sparsity big LLMs show. **→ C2 (contextual
  sparsity) is NOT worth building for the CP: even an optimistic 1e-2 threshold skips only ~14%,
  and that's before verifying it doesn't flip argmax. Negative result, settled.**

**Verdict (data-driven):**
- **C2 contextual sparsity — DROP.** CP FFN is dense; no exploitable sparsity.
- **C1 self-speculative — MARGINAL, conditionally.** Greedy spec-decode needs exact argmax match.
  q2 draft acceptance is hopeless (9%). int4 draft → int8/bf16 verify would accept ~the early
  codebooks (c1-c5, 57-77%) but reject the late ones, and you pay a full verify pass anyway, so
  net win is small and fragile. Best framing if pursued: int8 draft (78% vs bf16) with a single
  batched bf16 verify, accepting the common prefix of codebooks — but 78% per-codebook compounds
  to a low whole-frame acceptance. Re-derive expected speedup from these numbers before building.
- **A prosody knob — STRONGEST lead.** The late codebooks (c11-c15) ARE the texture/prosody
  control surface, and the Talker-feedback coupling means perturbing them shifts delivery
  globally. The q2-on-`down` "death metal" finding sits exactly here. A controllable strength
  knob on late-codebook CP precision is the most promising thing this measurement surfaced.

Caveat: teacher-forcing fixes the TOKEN feedback but each precision still builds its own intra-frame
KV cache, so per-codebook drift conflates step sensitivity + intra-frame KV accumulation (realistic,
but not a pure per-step isolate). Good enough for the decisions above.

**DECOMPOSITION EXPERIMENTS — ✅ DONE 2026-06-04 (commit pending; env knobs QWEN_TALKER_PREC /
QWEN_CP_LMHEAD_PREC / QWEN_CP_LAYER_PREC / QWEN_DUMP_CODE0; script `/tmp/ql_decomp.sh`).**
Tested the user's "low-bit early codebooks / high-bit late delicate ones" hybrid hypothesis. Verdict:
**the hybrid has NO sweet spot — the int4 cost is in the SHARED transformer, spread evenly.**

- **Talker int4 → code0 (= the WORDS) agreement 92.97% vs bf16** (teacher-forced). int4 on the Talker
  FLIPS ~7% of word tokens → confirms the workflow's #1 blind spot: **keep the Talker ≥ int8 for
  intelligibility.** (int8 on the 0.6B Talker is gated off at hidden<2048; needs 1.7B or gate bypass.)
- **Exp1 — drift is in the shared transformer, NOT the lm_heads.** Full int4 = 46%; int4-transformer +
  bf16-lm_heads = 48% (keeping heads precise recovers only +2%); **bf16-transformer + int4-lm_heads =
  84%** (the 15 lm_heads TOLERATE int4 fine). So the ONLY per-codebook weights (the heads) are not the
  problem → "keep late codebooks precise" = keep late heads precise = +2%, useless.
- **Exp2 — all 5 CP layers contribute ~equally** (one-layer-int4: L0 72%, L1 72%, L2 71%, L3 71%,
  L4 69%). No sacrificial layer; degradation compounds ≈multiplicatively (5 layers → 46%). Per-layer
  mix = smooth linear speed/quality trade, not a win.

**Consequences for the quant roadmap:**
- **int8 is the quality floor.** int4 on the CP transformer is unrecoverable via per-codebook (heads
  aren't the driver) or per-layer (all equal) tricks. DeepSeek-style per-tensor mix has no obvious
  CP sweet spot — the cost is in the shared FFN/attn.
- **Only safe low-bit hybrid: int4 on the 15 lm_heads** (84% agreement) → memory saving (60→15 MB),
  but heads are 4.3% of CP time → negligible speed. Marginal, clean, optional.
- [ ] **OPEN QUESTION (added 2026-06-04): can int4 shed the "anger" and become the TRUE default, or not?**
  int4 introduced slight aggression (ear), q2 strong "death-metal" — the quant-ladder localised it to
  the LATE/fine CP codebooks (c11-c15: int4 23-27% vs bf16) that carry texture/prosody. The anger is
  the SAME perturbation as the prosody knob (track A), just uncontrolled. Question: is there an int4
  variant that keeps the speed/memory but NOT the roughness? Candidates to test: (a) **int4 transformer
  + int8 (or bf16) lm_heads** — but Exp1 showed heads aren't the driver (+2%), so unlikely; (b) **finer
  int4 (per-group scales / smaller block than 32, or int4 + a few high-precision outlier channels** à la
  AWQ/per-channel) — the block-32 absmax may be too coarse for the sensitive late residuals; (c) **int4
  everywhere EXCEPT the FFN `down` (kept int8/bf16)** — `down` is the causal driver of the roughness
  (q2-on-down = death-metal), so int8-down + int4-rest might remove the anger at most of the speed. If
  any (b)/(c) yields int4-speed + int8-quality → int4 becomes the real default; if not, int8 stays the
  default and int4 lives only as the deliberate `--roughness` knob. MEASURE with the quant-ladder (code0
  + per-codebook agreement) AND by ear. Decide the `.qvoice`-save-quantized design (test backlog) after.
- **Speed must come from compute throughput, not byte-cutting:** the CP is COMPUTE-bound after
  SDOT (the workflow corrected our "bandwidth-bound" framing — int8 weight → one vdotq_s32 vs
  dequant+FMA×8, so the FMA reduction is the limiter; int4's 1-row q4 kernel made x-loads the new
  bottleneck = why int4 lost). Levers: SDOT (done, ARM), VNNI (x86, written/unvalidated), SDOT on
  `argmax_matvec_int8` (CP lm_heads still on f32 dequant → easy ~−20% on the 4.3% slice), 2-row-fuse
  the q4_0 kernel (separate the architecture penalty from the −26% true quality floor).

---

## End-to-end leverage map (workflow `qwen-tts-flow-map`, 6 agents, 2026-06-04)

A multi-agent static map of the whole pipeline (Talker / CP / decoder / conditioning / kernels),
spot-verified against source. Condensed; full synthesis in session transcript.

**Byte & time budget per frame (12.5 Hz → 80 ms audio/frame, 0.6B/M1):**
| Stage | re-reads/frame | time share | bound |
|---|---|---|---|
| **Code Predictor** | 30 MB transformer re-read **15×** + 15 lm_heads 1× | **74–90% (~76 ms/f)** | bandwidth at bf16 → **compute at int8+SDOT** |
| Talker (code0=words) | 28 layers 1×; codec_head 1×; KV pos+1 | 9–15% (~25 ms/f) | compute |
| Speech decoder | conv ~80 MB f32 1× on drain | 5–8% non-stream, ~0% streamed (overlapped pthread) | conv-bound, off critical path |
| Conditioning/prompt | prefill-amortized | negligible/frame | — |

**Quant-sensitivity ranking (most → least fragile):** (1) Talker code0 path (codec_head bf16 but its
input quantized — HIGH, now measured: int4 flips ~7% words); (2) CP late codebooks c11-15 (HIGH,
measured); (3) CP FFN gate_up/down = 53% of CP time + the causal driver of per-codebook drift + the
q2 "roughness" effect (HIGH); (4) CP attn q/k/v/o (MEDIUM); (5) Talker bf16 KV (MEDIUM); (6) decoder
ConvNeXt pw1/pw2 if ever quantized (MEDIUM-HIGH, currently f32); (7) embeddings + all norms (LOW —
keep f32/bf16). **codec_head itself is NEVER quantized despite gating intelligibility.**

**Speed levers (ranked):** SDOT (validated, ARM) · **0.6B Talker int8 — ✅ SHIPPED 2026-06-04**
(dropped the stale `hidden<2048` gate; --int8 now quantizes the 0.6B Talker too: Talker step
30→16 ms/f = −47%, 0.6B int8 RTF → 0.87–0.92 sub-1.0, no 4-thread hang, code0 96.9% = int8-gentle on
words, user ear-approved, int8 golden regenerated) · **x86 VNNI + cross-OS pool (written, UNVALIDATED —
x86 is scalar+single-thread today, the #1 unblock)** · server continuous-batching (throughput only).
**Refuted/closed (all verified 2026-06-04):**
- int4-Q4_0 global on CP, self-speculative (marginal), contextual sparsity (CP FFN dense).
- SDOT-on-lm_head — forks int8 trajectory (golden mel-corr → 0.51); reverted.
- q4_0 2-row-fuse — no win on NEON (decode-bound, not x-load-bound); reverted.
- **"vectorize RoPE + attention" — ALREADY DONE (workflow was WRONG here).** Hot-path attention
  (`qwen_causal_attention_bf16kv`) has full NEON (Q·K dot + online-softmax V accumulation) + AVX2
  twins; RoPE (`apply_rope_neox_inplace`, Talker + CP) is NEON+AVX2. The only scalar RoPE
  (`qwen_apply_rope_interleaved`, kernels.c:2209) has ZERO callers = dead code. Nothing to do.
- **"fuse codec_head into final Talker wo" — NOT a real fusion.** codec_head applies to last_hidden
  = output AFTER the final layer's residual+FFN+RMSNorm (nonlinearities in between), so it can't fold
  into wo. It's one already-optimized bf16 matvec (~1% of total). Skip-special-token rows would save
  ~0.3% total — not worth the branching. Closed.

**Prosody/instruct knobs (ranked, concrete):** (1) `--roughness` = bf16↔q2 blend on CP `down`
(code_predictor.c FFN); (2) steerable prosody vector added to `cp_x` BEFORE CP layer 0 (the single
Talker→CP injection point, `cp_mtp_project`); (3) instruct→CP control vector `diff(neutral−angry)`
injected scaled (amplifies the architecturally-weak instruct); (4) per-codebook weights on the CP-code
feedback sum into the next Talker step (`qwen_tts.c:1300`, currently flat sum); (5) speaker_scale on the
voice-clone ECAPA norm-match; (6) per-step instruct re-injection decayed across codebook steps.

**Open questions still needing measurement:** intrinsic-vs-KV-accumulation isolate (bf16 KV +
quantized matvec only); `--roughness`/q2 generalization across speakers×langs×models;
do CP layer-wise emotion directions exist (dump neutral-vs-angry activations); x86/VNNI runtime
correctness + RTF (Ryzen box); 2-row q4_0 true quality floor; decoder bf16 (cheapest unmeasured
DRAM saving, 80 MB f32 → no quant infra yet).

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
