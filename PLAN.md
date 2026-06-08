# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-06-04

Core engine is **COMPLETE** and producing good audio for both 0.6B and 1.7B.
INT8 is validated end-to-end. **x86 is no longer Apple-only:** AVX2 hot-path twins +
cross-OS pthread pool + int4 are **validated on a Ryzen 7 6800H (2026-06-04, RTF ~2.02)**.
Remaining x86 work = **AVX-512/VNNI** (code written, needs a rented Zen4+/Intel box; the
6800H has no AVX-512). The current focus is **hybrid mixed-precision quant**, the
**prosody/roughness knob** (the differentiator), and the test/release backlog. History is
compacted at the bottom — nothing dropped.

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

> ⚠️ **SDOT is ARM-only (`vdotq_s32`).** Its x86 equivalents are now WRITTEN (2026-06-04):
> the **AVX2 int8/bf16 matvec twins** (`#elif __AVX2__`, validated on the Ryzen box — see 21.3)
> and the **AVX-512 VNNI** native int8 dot (`_mm512_dpbusd_epi32`, `SIMD=avx512vnni`, commit
> d67648a — **UNVALIDATED**, needs a Zen4+/Intel AVX-512 VPS; the Ryzen 6800H is AVX2-only).
> `quantize_act_int8` is portable C, reused as-is by the VNNI path. **x86 reality now (Ryzen
> 6800H, validated): AVX2 + pthread pool + int4 → RTF ~2.02 on 0.6B** (memory-bound, AVX2 only
> ~6% over scalar; int4 multi-threaded is the real lever). The "x86 is all scalar / AVX file
> empty" claim is OBSOLETE.
- [ ] `[LOW]` No `silvio_17b.qvoice` to test custom voice on 1.7B (needs `qwen3-tts-1.7b-base` to
  create). 1.7B preset already validates the Talker int8+SDOT critical path.

> **Gate status (updated 2026-06-04):** ① x86 re-audit DONE (deep re-read, not a grep — confirmed
> AVX2 existed only on 5 aux ops; now AVX2 twins written for all hot ops). ② `silvio_17b.qvoice`
> created + 1.7B custom voice validated. ③ `feat/int8-sdot` already MERGED → `feat/labs` (14a0239).
> Branch `feat/avx2-xos-threading` carries the AVX2/threading/int4/VNNI work, NOT yet merged.
> **Still owed before a release:** the test backlog in 21.6 (qvoice×quant matrix, TSan race,
> server RTF re-run) + AVX-512/VNNI validation on a rented box.
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
  NeoX RoPE in talker/code_predictor/speech_decoder, and snake. **✅ RUNTIME-VALIDATED 2026-06-04
  on the Ryzen 7 6800H** (Zen3+, WSL2): `make blas` builds clean with portable `-mavx2`, `--caps`
  reports `AVX2 (2-row fused, FMA)` + pthread pool, output coherent by ear. **Measured (0.6B,
  4-thread): AVX2 only ~6% over `SIMD=scalar`** (hot path is DRAM-bandwidth-bound, not compute) —
  so the SIMD twin is correct but not a miracle; **int4 multi-threaded is the real x86 lever: RTF
  2.81→2.02 (−28%)**. Full numbers in `docs/building.md` "Performance notes (x86 / WSL2)".
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
- [~] `[MED]` **VNNI** = x86 SDOT twin for int8 — **WRITTEN 2026-06-04 (commit d67648a),
  UNVALIDATED.** `_mm512_dpbusd_epi32` AVX-512-VNNI path, `SIMD=avx512vnni` build
  (`-mavx512f -mavx512bw -mavx512vl -mavx512vnni`), `--caps` reports `int8 dot: VNNI (native)`.
  `quantize_act_int8` reused as-is. **Can't run on the Ryzen 6800H (AVX2-only, no AVX-512)** →
  next x86 step is to **rent a Zen4+/Intel AVX-512 VPS**. **VALIDATION TOOLING READY (2026-06-05):**
  `./qwen_tts --self-test` + `make test-selftest` = a model-free kernel numeric gate (matvec
  bf16/int8/argmax vs f32 reference, `qwen_kernel_selftest` in kernels.c) — **immune to the greedy
  trajectory fork that makes cross-ISA `test-golden` a false alarm** (the right way to prove VNNI:
  correct → int8 rel_L2 ~4e-3, broken offset → blows up). One-shot `bash tests/vps_validate.sh
  [model]` drives the whole VPS run (cpuinfo check, build avx512vnni, `--caps` asserts VNNI,
  self-test, int8/int4 ×{-j1,-j4} RTF, VNNI-on-vs-off A/B). Validated on M1 (SDOT path rel_L2
  3.8e-3, fallback 1.8e-7, both PASS). Verify ISA with Intel SDE in CI before trusting perf on HW.
- [~] `[LOW]` **AVX-512 bf16 matvec** (`__m512` 16-wide) — **WRITTEN 2026-06-04 (commit b89f30e),
  UNVALIDATED**, `SIMD=avx512` build. Same rented-box gate. (Note: this is the `__m512` widen-FMA
  path, not yet the native `_mm512_dpbf16_ps` AVX512-BF16 dot — that's a further upgrade.)
- **Order**: ARM NEON headroom (21.3b, dev HW measurable now); x86 AVX2+threading+int4 ✅ DONE on
  the Ryzen box; **only AVX-512/VNNI remains → rented Zen4+/Intel VPS** (can't run AVX-512 on M1
  or the 6800H).

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
- [ ] `[LOW/RESEARCH]` **SME/SME2** (Apple M4, ARMv9.2) — scalable matrix extension; the GEMV→GEMM
  batching path (track B) could map onto it. Only on M4+; research after bf16/i8mm.

> 💡 **Scaleway rents Apple Silicon Mac mini (M1/M2/M3/M4) by the hour** (account already set up).
> This is the **ARM twin of the x86-VPS workflow** we just used for AVX-512/VNNI: rent a newer Mac
> mini for a few hours, implement+validate the post-M1 NEON twins on real silicon, then shut down.
> - **M2/M3** (ARMv8.6: `__ARM_FEATURE_BF16` + `__ARM_FEATURE_MATMUL_INT8`) → validate `vbfdot`
>   (bf16 dot) and `smmla` (i8mm, ~2× on int8 matvecs) — the two `[MED]` items above. Dev M1 can't
>   test these (lacks the features); they're written behind `#if __ARM_FEATURE_*` with NEON fallback.
> - **M4** (ARMv9.2) → also SME/SME2. Likely back-compatible (runs the M1 NEON+SDOT path unchanged),
>   so it doubles as a "does v0.9.0 still scream on the newest Apple chip" RTF check.
> - Reuse `tests/x86_bench.sh`'s spirit: a small `bash tests/arm_bench.sh` (TODO) + `--caps` +
>   `--self-test` (the self-test is ISA-agnostic → it already gates a new ARM kernel for correctness).
> - Validate each new ARM kernel with `make test-golden` (same-ISA) + `--self-test` before trusting.

### 21.3c Non-Apple ARM — are we safe? (the same question x86 answered)

**Short answer: the code is SAFE and should run correctly on any aarch64, but it's UNVALIDATED on
non-Apple ARM and leaves the newer SIMD on the table** — exactly the position x86 was in before the
Ryzen/EPYC runs.

- **Baseline NEON is universal on aarch64** (`#ifdef __ARM_NEON`, present on every ARMv8) → the
  matvec/attention hot path runs on Graviton, Ampere Altra, Snapdragon, NVIDIA Grace/GB10, etc.
- **SDOT** is `#if __ARM_FEATURE_DOTPROD` with a NEON fallback → present on Graviton2+/Ampere/
  ARMv8.2+ (graceful step-down if absent). **Threading**: the cross-OS pthread pool (21.2) already
  gives Linux ARM multi-thread (was single before). So nothing *breaks* off-Apple.
- **GAP 1 — never benchmarked on non-Apple ARM** (same gap x86 had). The RTF numbers are Apple-M1-only.
- **GAP 2 — `-march=native` build portability** (the ARM mirror of the x86 SIGILL fix): a binary
  built with `-march=native` on a DOTPROD/i8mm-capable host SIGILLs on an older ARM core. We added the
  x86 runtime ISA guard + portable `-mavx2` default; **ARM needs the same** (portable baseline +
  `qwen_check_runtime_isa()` for ARM features + runtime step-down i8mm/bf16→SDOT→NEON→scalar).
- [ ] `[MED]` **Validate the NEON+SDOT path on a non-Apple ARM box.** Scaleway has **COPARM = Ampere
  Altra** (Neoverse-N1, ARMv8.2, has DOTPROD) — rent it, `--caps`/`--self-test`/`test-golden` + RTF,
  same flow as the EPYC run. Confirms "ARM-everywhere" the way the EPYC confirmed "x86-everywhere".
- [ ] `[LOW]` **ARM runtime ISA guard + portable build** (GAP 2 above) for a shippable non-native ARM binary.
- **NVIDIA DGX Spark (GB10)** is a concrete future target: 20× Arm Neoverse (ARMv9.2 → NEON+SDOT+**i8mm+
  bf16+SVE2** all present) + 128 GB unified LPDDR5X. Our **CPU path runs correctly today** (NEON+SDOT+
  pthread) but leaves i8mm/bf16/SVE2 unused; the unified memory is M1-like (good for the 16×-reread →
  possibly strong RTF). The GB10 **Blackwell GPU is out of scope** (would need a CUDA backend — separate
  large effort). So on a DGX Spark we're "safe + correct on CPU, with headroom", not "optimal".

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
| **Linux ARM (aarch64, M1-class)** | NEON ✅ | NEON ✅ | pthread pool ✅ (2026-06-03) | correct + multi-thread; bf16/i8mm NEON headroom remains |
| **Linux/WSL/Win x86-64** | **AVX2 ✅** (all matvec/attn, 2026-06-04) + VNNI written/unvalidated | **AVX2 ✅** | pthread pool ✅ | **Ryzen-validated: RTF ~2.02 (int4, 4-thread); bandwidth-bound so AVX2 only +6% vs scalar** |

**Coverage matrix we OWE users** (goal: NEON-equivalent everywhere SIMD exists, scalar always):
| Op family | NEON | AVX2 | AVX512/VNNI | scalar | Gap |
|---|---|---|---|---|---|
| matvec bf16/int8/q4_0 + argmax | ✅ | ✅ (2026-06-04) | written, **unvalidated** (rented box) | ✅ | bf16/i8mm NEON (21.3b); AVX-512 run on a real box |
| attention (3 variants) | ✅ | ✅ (2026-06-04) | n/a | ✅ | — |
| RoPE / bf16 pack / swiglu / add·mul·scale / snake | ✅ | swiglu/add·mul·scale ✅; RoPE/bf16-pack/snake still scalar | n/a | ✅ | AVX2 the remaining inline NEON-only ops (low priority, off hot path) |
| rms_norm ×3 / bf16_accum / bf16_to_f32 | ✅ | ✅ | ❌ | ✅ | AVX512 optional |

Status: 21.2 (threading) ✅ all platforms · 21.3 (AVX2 on all hot ops) ✅ validated on Ryzen ·
AVX-512/VNNI written but needs a Zen4+/Intel box · 21.3b (post-M1 NEON) still open.
**RTF numbers now exist for x86 too** (Ryzen 6800H: int4/4-thread ~2.02), not Apple-M1-only.

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

- [x] **Concurrent server worker pool — IMPLEMENTED (2026-06-04, branch feat/avx2-xos-threading).**
  The first real step of the "batching" feature: request-level parallelism (NOT yet continuous
  batching of matvecs — see §B). `--workers N` (default 1 = unchanged behavior). For N≥2:
  acceptor thread + bounded connection queue (256) + N worker threads; worker 0 reuses the base
  ctx, the rest are **`qwen_tts_clone_for_worker()`** clones that SHARE the read-only weights +
  loaded voice + RoPE (weight memory paid once) but own FRESH per-request mutable buffers (KV
  caches, dec_*/cp_dec_*/pref_*, emb LRU cache, delta-prefill cache). Freed with
  `qwen_tts_free_clone` (never `qwen_tts_unload` — that would free the shared weights).
  THREE shared-mutable hazards found + fixed: (1) the synthesis ctx → per-worker clones;
  (2) the kernel thread pool has a single global job slot → added `qwen_parallel_is_reentrant()`
  (**GCD = safe** for concurrent `dispatch_apply`; **pthread/Win32 = NOT** — one `P.job`), so the
  server serializes synthesis under `g_synth_lock` ONLY when the pool is non-reentrant
  (`g_serialize_synth`); on GCD it runs fully parallel; (3) the RNG `g_seed` was a global →
  made `__thread` (was a cross-request race + broke per-request seed reproducibility). Also
  `inet_ntop` replaces thread-unsafe `inet_ntoa`. **VALIDATED on M1 (GCD): 2 workers, 2 concurrent
  requests, both bit-identical to the single-worker reference (mel-corr 1.00000) across
  {bf16, int8, int4, voice silvio_06b+int8}; concurrent timing ~equal (true overlap, not
  serialized); clean kill-by-name teardown, no orphans.** `make test-serve-concurrent` +
  `tests/test_parallel.sh`. ⚠ On the pthread/Win32 backend (the VPS) concurrent synthesis is
  CORRECT but serialized (no overlap) until the pool is made reentrant — tracked below.
  ⚠ Memory: clones share weights, so extra cost ≈ N×(KV + work buffers), NOT N×model.

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
- **NEW concurrency tests (for the batching work):**
  - [x] (a) **parallel-request test — DONE** (`make test-serve-concurrent`, `tests/test_parallel.sh`):
    2 concurrent curls at a `--workers 2` server, each output compared to a single-worker reference via
    mel-corr (≥0.98), across {bf16, int8, int4, voice+int8}. All corr=1.00000 on M1. Timeout+pkill harness.
  - [ ] (b) **race detector — `make test-serve-race` under TSan** (`-fsanitize=thread`) + concurrent load.
    Now MORE valuable: with the worker pool live there IS real concurrency to instrument. Must confirm
    (i) per-worker clones never alias, (ii) on GCD the no-lock path is genuinely race-free, (iii) the
    `g_serialize_synth` lock path is correct on the pthread backend. Build a TSan target (ASan/TSan are
    mutually exclusive — separate build).
  - [ ] (c) **streaming RTF + verify under int8/int4** (TTFA + RTF per quant), AND streaming under
    `--workers ≥2` (the chunked-stream path holds the connection open longer → exercises the queue/worker
    lifecycle differently than full-WAV). All must use the timeout+pkill harness (runaway lesson).
- [ ] **Make the pthread/Win32 kernel pool reentrant** so off-Mac servers get TRUE `--workers` parallelism
  (today serialized via `g_serialize_synth`). Options: per-submitter job slots, or a pool-of-pools. Verify
  on the Ryzen VPS. Until then x86 concurrent serving is correct-but-serial.

**PERFORMANCE / PORTABILITY (largely CLOSED 2026-06-04 — see 21.2/21.3/21.3b):**
- [x] **All 8 hot matvec/attention kernels now have AVX2 twins** + cross-OS pthread pool (was
  scalar+GCD-only). ✅ Ryzen-validated. Remaining scalar-on-x86: SwiGLU `expf` (`kernels.c:1433`,
  macOS uses Accelerate `vvexpf`), inline RoPE/bf16-pack/snake — all low-priority/off hot path.
- [ ] `[LOW]` post-M1 NEON headroom (bf16 vbfdot / i8mm smmla — 21.3b).
- [x] **Makefile `-march=native` off-Mac → portable `-mavx2` + runtime ISA guard** (no SIGILL).
  Still missing: true per-CPU function-multiversioning (one fat binary stepping SSE→AVX2→AVX512).
- [ ] **CI is build-only off-Mac** — `.github/workflows` build Linux x86/ARM + macOS-ARM but run only
  `./qwen_tts --help`; **no inference is ever executed off Apple Silicon, and x86 has NEVER been
  benchmarked.** No macOS-x86, no Windows. Add a real inference smoke (small model) + SDE for ISA.
- [ ] **Windows native won't compile** (`mmap`/`pthread`/`gettimeofday`, no Win32 fallback) — WSL2 only.

> ✅ **x86 AVX2 + threading VALIDATED on the Ryzen 7 6800H (2026-06-04)** — the test that "never
> happened before". Built under WSL2 with portable `-mavx2`, `--caps` confirms AVX2 + pthread pool,
> output coherent. Findings: hot path is DRAM-bandwidth-bound → AVX2 only +6% over scalar; **int4
> multi-threaded is the x86 lever (RTF 2.81→2.02, −28%)**; 4 threads sweet spot (8 regresses); High-
> performance Windows power plan matters. Full writeup in `docs/building.md`. **STILL OWED: AVX-512/
> VNNI** (code written, 6800H is AVX-512-less) → **rent a Zen4+/Intel VPS** more powerful than the
> mini PC, build `SIMD=avx512vnni`, validate golden mel-corr + measure RTF (Intel SDE in CI first).

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
  md5)**: md5 flakes even at `-j1 temp0` (±1 LSB decoder noise, verified). **⚠ CORRECTION (Ryzen
  2026-06-04): mel-corr is NOT a valid CROSS-ISA gate.** Greedy `-j1 temp0` decode forks the whole
  trajectory on a single epsilon-different logit across ISAs → x86 golden scored mel 0.55–0.85 vs the
  ARM golden despite producing perfectly coherent speech (confirmed by ear). The old "must stay 0.99+
  cross-ISA" assumption is NAIVE for greedy autoregression — retracted. Same-ISA it's still the safety
  net. **The real cross-ISA correctness gate is now `make test-selftest`** (kernel numeric, below).
- [x] **`test-selftest`** (2026-06-05, `qwen_kernel_selftest` + `--self-test`) — **the cross-ISA gate
  that test-golden can't be.** Compares the dispatched matvecs (bf16/int8/argmax-int8) to an f32
  reference on deterministic random data → catches a broken SIMD kernel (esp. the AVX-512/VNNI int8
  dot + `__m512` bf16 matvec) WITHOUT running the pipeline, so the greedy trajectory fork can't mask
  it. Runs the dispatched path AND the scalar/widen fallback (`QWEN_NO_SDOT/QWEN_NO_VNNI`). int8 uses
  a near-zero-robust L2-relative metric (act-quant → ~4e-3 correct, blows up if broken). Wired into
  `test-all`; `tests/vps_validate.sh` drives the full AVX-512 VPS run around it.
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

### A. Prosody/emotion control on the Code Predictor — BUILT v1 (`feat/expressivity`), NOT YET AT THE TOP

> **▶▶ SESSION 2026-06-07 WRAP (read FIRST — 12 commits `3e0cb70`→`c1c4d2b` on `feat/expressivity`, NOT merged).**
> Turned the half-done emotion manifest into a full expressive-control system, all ear-validated with the user.
> SHIPPED THIS SESSION:
> - **Compound-emotion MANIFEST** (`qwen_tts_emotion.{c,h}`): `--emotion <mood>` sets the whole recipe
>   {vec, steer_weight, roughness, volume, rate}. Explicit flags override. Language-aware resolver (IT→`it_centered/`).
> - **`--volume` (PCM gain) + `--rate` (in-engine WSOLA, pitch-preserving, no ffmpeg)** in `qwen_tts_audio.c`.
> - **`--compose` + INLINE `[tag]` MARKUP** (ElevenLabs/Bark style, English tags, auto-detected in `--text`): per-span
>   emotion switch, `[pause:400ms]`, paralinguistic fillers. `parse_markup`→`render_spans`. docs/markup.md.
> - **🐛 CROSS-SPAN LEAK FIXED (user-caught by ear):** compose's repeated `generate` calls reused the prev span's KV
>   via delta-prefill → a `[sad]` rendered differently in a mix vs alone. Fix `ctx->prev_prefill_len=0` per span →
>   each span bit-identical to standalone (corr 1.0). LESSON: delta-prefill is a SERVER opt, wrong for compose.
> - **PARALINGUISTIC LIBRARY** (ear-validated macros, no-steer soft prosody): `[sigh]`=Hah…, `[hmm]`=Hmmm…, `[mmm]`=嗯,
>   `[mah]`,`[uhm]`,`[hmpf]`,`[ahh]`/`[relief]`=Haaa…,`[phew]`=Uao…,`[laugh]`=Eheh…(IT),`[haha]`(EN),`[heh]`,
>   `[ouch]`(EN)/`[ahi]`(IT). `tests/sound_suite.sh` = the discovery workflow (mass-gen→listen→bake; CLI per line).
> KEY DISCOVERIES (validated): `Hah...`=sigh / `Hah`=laugh (the "…" makes the breathy sigh; bare = a dry laugh);
> **Chinese phonetic chars are a clean sound source even under IT**: 哈哈/嘿嘿/呵呵=laughs, 唉=weary sigh, 嗯=mmm;
> 😂 leaks a faint sigh; "Ahahah 😆"→"AHI!" pain. RULES baked in: macro rate ≥0.90 (slower=metallic WSOLA on a short
> vowel); a trailing "…" can trigger a 2nd spurious vocalization; same string→different sound per language; **weight is
> an UP-mood lever only — down-moods (sad/gloomy) go off-manifold ('Chinese tone') at high weight → use LOW steer +
> prosody** (sad=w1.1/rate1.08/vol0.86). Golden mel-corr still 1.0 (default-off untouched); `make test-emotion`/`test-compose` green.
> NEXT (ranked): (1) reply to Leo (perf, task #3, prepped); (2) run sound_suite + bake more winners (CN laughs, pain
> 'ahi', `[ha]`=dry laugh); (3) **RE-TEST on a CLONED `.qvoice` (Galatea/Silvio): emotion-MIX (`[happy]/[sad]/[angry]`)
> + paralinguistic fillers (`[sigh]`/`[laugh]`/`[mmm]`) via `--compose`/markup — all validation so far is on PRESET
> ryan; verify the macros + mid-text emotion switches survive on a cloned voice (CP is cross-model so it SHOULD, but
> the no-steer fillers + per-span cold-prefill need ear-checking on a real clone)**; (4) server-side volume/rate/compose
> (CLI-only today); (5) merge feat/expressivity→feat/labs.
> KNOWN: server doesn't have volume/rate/compose; true breaths absent (no <breath> token); model occasionally
> over-elongates short words (intermittent, model-side). Local-only (not git): `samples/` (all audio), `voices/galatea_06b.qvoice`.
> Stray `analisi_leak*.md` still untracked (NOT ours). BACKLOG below: int2/int3 quant, clone+preset speed/quality bench, emoji map.
>
> **▶ RESUME 2026-06-07 (earlier in session — kept for detail; see `docs/expressivity-recipes.md`).**
> Session 2026-06-06 was long & productive. WHERE WE LANDED:
> - **Galatea = the IT demo voice** (`voices/galatea_06b.qvoice`, CC LibriVox, better than Silvio). Centered IT
>   palette (`presets/emotions/it_centered/`) = the one to use (collinearity fixed, +88% contrast).
> - **Validated recipe doc committed: `docs/expressivity-recipes.md`** (joy=excited NOT happy, sad=slow+pauses,
>   annoyed=angry+roughness, news=proud; per-language map; dead-ends; emergent-basin map + the ES "eeem"/sbuffo reproducer).
> - **Settled with proof:** think=language-slot (our RE correct, vs official source); instruct is Chinese-tuned/weak-on-EU
>   & not exposed for clones → CP-steering is the only emotion path for cloned EU voices. Closed dead-ends: full rage,
>   paralinguistic tags, relax-identity (`QWEN_SPK_SCALE` knob added, no-op default), real-breath splice, DSP pauses, hesitant-capture.
> - **`--volume`/`--rate` flags = trivial future add** (volume=pure PCM gain; not yet exposed).
> NEXT STEPS (ranked): (1) **reply to Leo** (task #3 — perf leads int8-Talker/Windows-SD-single-thread/9P + his 2 forked
>   params `--emit-tokens`/`--decode-tokens` deduced; re-read his full email first).
>   (2) ✅ **DONE 2026-06-07 (commit 3e0cb70): compound-emotion MANIFEST as code** — `qwen_tts_emotion.{c,h}`,
>   13 moods `name → {vec, steer_weight, roughness, volume, rate}`; `--emotion joy/sad/stern/annoyed` sets ALL knobs.
>   Also shipped: `--volume` (PCM gain) + `--rate` (in-engine WSOLA time-stretch, no ffmpeg) + language-aware
>   resolver (Italian auto-uses `it_centered/`) + graceful degrade when a mood's vec is missing for a language.
>   `make test-emotion` (in test-all); golden mel-corr still 1.0 (default-off).
>   **EAR-VALIDATED + RECALIBRATED 2026-06-07 (ec30e51):** sad@2.0 went off-manifold ('Chinese tone') →
>   down-moods = LOW steer + strong prosody (sad=w1.1/rate0.80/vol0.86 = 'sad+slight ache', user 'ottimo'; gloomy=w0.5).
>   (2b) ✅ **DONE 2026-06-07 (bbec95f + 17ca50a): --compose + INLINE MARKUP for audiobooks.** Per-span synthesis
>   (qwen_apply_emotion re-entrant + render_spans + parse_markup), ElevenLabs/Bark-style English square-bracket tags
>   auto-detected in --text: `[sad]/[excited]` mid-text emotion, `[sigh]/[huff]/[ugh]/[groan]/[hmm]` paralinguistic
>   fillers (generated NOT spliced — the slow+sad recipe elongates "Ehh.."/"Uff.." vowels), `[pause:400ms]/[break:1s]`.
>   Spans model-generated → seamless concat (NOT the dead-end ref-splice). `make test-compose`; docs/markup.md.
>   KEY SERENDIPITY: slow+sad config = a controllable paralinguistic GENERATOR (user idea). STILL TODO: server-side
>   volume/rate/compose (CLI-only); true breaths still absent; tune macro recipes; EN/other-lang sigh quality.
> (3) optional: dedicated ES palette capture (ES inverts everything); retry sbuffo at lower weight; more macros.
>
> ## ★ BATCHING ARCHITECTURE — TWO PRODUCTS, ONE ENGINE (vision, user 2026-06-08)
>
> The batched compute kernels (ragged Talker step + per-stream sampling/EOS + batched CP + per-seq
> prefill — **DONE**) power **two distinct, both-wanted products**. Same engine (~90% shared), slightly
> different use cases. KEEP BOTH.
>
> **PRODUCT 1 — `--batch` LONG-FORM (shipped, this branch).** One user, ONE long text (podcast, audiobook,
> long article). The CLI splits it into sentence-packed chunks, steps them TOGETHER (weight-stationary),
> and re-stitches one continuous WAV. A **productivity / throughput flow for long content**: a paragraph
> that took 10 min single-stream finishes in ~half. Decode is post-hoc (whole text), so it does NOT stream
> — and that's fine, it's a batch-job lever. bf16 1.65–1.74× on M1; int8/int4 supported. THIS IS DONE and we
> WANT to keep it — its use case (long-form authoring) is real and distinct from serving.
>
> **PRODUCT 2 — SERVER REQUEST-BATCHING (the big feature, dedicated branch, vLLM-style).** N DIFFERENT users,
> N DIFFERENT short requests, served concurrently with **maximum efficiency**. Reuses the SAME batched
> kernels, but the B sequences are different users' requests (not chunks of one text), and a **continuous-
> batching scheduler** keeps the in-flight batch full (admit a new request into the slot a finished one frees
> — not static batching, which wastes utilization on ragged EOS). **Streaming COMPOSES here**: the batch steps
> one frame at a time → after each batched step every active request has a new frame → emit it to THAT
> request's SSE while the Talker+CP compute stays batched → **real parallel streaming to N users** (exactly how
> vLLM streams tokens). Throughput (batched compute, shared) + latency (per-request streaming decoder + SSE)
> TOGETHER; good per-request TTFA (prefill + admit in-flight). This is the production-serving architecture.
> What's missing = scheduler + concurrent server + per-request streaming decode (see the SERVER REQUEST-
> BATCHING task below). Pays most on bandwidth-bound x86 (EPYC/Sapphire) → build correctness on M1, validate
> perf on a rented VPS.
>
> **One-line distinction to repeat everywhere:** `--batch` = split ONE long text for ONE user (audiobook/
> podcast throughput, no streaming, shipped); SERVER batching = serve N users' DIFFERENT requests at once
> (max-efficiency serving, real parallel streaming, to build). Same batched-compute foundation.
>
> **BACKLOG / IDEAS TO ANALYZE+VALIDATE (user, 2026-06-07):**
> - **BATCHING (branch `feat/batching` off feat/expressivity, started 2026-06-07):** OPT-IN alternative path
>   (vLLM-style — default = today's single-stream, untouched, golden bit-identical). Premise TESTED via
>   `make batching-bench` (`tests/batching_bench.c`, docs/batching.md): batched GEMM(16) vs 16× GEMV →
>   **~2× on M1 at bf16** (single-stream is compute-bound there, the FLOOR); **batching is worth MORE at lower
>   precision** (int4/int2 amortize the costly per-token nibble/bit UNPACK) → **pair batching with int4/int8, not bf16**;
>   could even make int4 viable on M1 (where nibble-unpack is the slowness today). Cross-CPU prediction: every x86 box
>   ≥ M1 (Ryzen mini-PC >2×+int4 sweet-spot; EPYC Turin VPS ~2-4×/core × core-scaling = the throughput play; Zen5+VNNI
>   best target). VALIDATION ORDER: M1 → Ryzen mini-PC (Zen2/3, RDP) → Turin VPS (AVX-512) → ONLY THEN discuss w/ Leo.
>   Full prototype scope in docs/batching.md (B per-seq KV, GEMM step kernels Talker+CP, ragged-EOS, chunk scheduler,
>   reuse render_spans for concat). Commits on feat/batching: d175eb1, 5027490, 190a8f7, f34766d, +.
>   **NON-NEGOTIABLE CONSTRAINTS (user 2026-06-07):** (1) ADDITIVE flow — `if(--batch){new}else{exactly as today}`;
>   NEVER zap/rewrite the working single-stream path; reuse where possible, new code for batched parts; golden bit-identical.
>   (2) MULTI-ISA always: every batched kernel = NEON + AVX2 + AVX-512 + scalar (not NEON-only). (3) Annotate newer-ISA
>   leads: ARM bf16 BFDOT/BFMMLA + i8mm SMMLA (M2/M3/M4/M5, Neoverse V1/V2, NVIDIA Grace/DGX Spark), SVE/SVE2, x86
>   AVX-512-BF16/VNNI; add int8/int4 batched twins (batching pays most at low precision).
>   DONE so far: (a) `qwen_matmat_bf16` batched step primitive, multi-ISA NEON/AVX2/AVX512/scalar, --self-test PASS
>   (matmat(B=8) vs B×matvec L2_rel ~6e-7); (b) **`qwen_batch_talker_step` — batched Talker step (opt-in), commit 6851eb0**:
>   B seqs in lockstep through the full Talker (B per-seq KV + batched attention), REUSES per-vector kernels (rmsnorm/
>   rope/attn/swiglu looped over B), batches ONLY the matvecs (gather→matmat→scatter). `--batch-test`/`make test-batch`:
>   **WIRING bit-EXACT vs single-stream** (force_matvec mode L2_rel 0.00) + matmat-kernel probe 6e-7. **KEY FINDING:**
>   the real matmat path diverges ~1.6% in HIDDEN state purely by fp accumulation ORDER amplified through 28 layers
>   (1.46×/layer) — a valid alternative kernel LIKE INT8, NOT a bug; validate end-to-end by AUDIO mel-corr, not hidden
>   bit-match. v1 = bf16, lockstep (no ragged EOS). (c) **`qwen_batch_cp_predict` — batched Code Predictor (the
>   90%/frame bottleneck), commit 09edb01**: B frames lockstep through the 16-step CP (B per-frame CP KV), reuses
>   CP layer math, batches matvecs via shared `qwen_batch_proj`; steering supported, v1 bf16 no-roughness.
>   `--batch-test`: **CP wiring 0/120 codes differ (bit-exact) AND CP matmat 0/120 differ** — greedy argmax absorbs
>   the fp-order noise → batched CP yields IDENTICAL audio codes. So both batched COMPUTE kernels (Talker+CP) are
>   built + validated; golden 1.0 (additive).
>   **⚠️→✅ FINDING REVERSED (2026-06-08): batching WINS ~2× on M1 once the matmat is register-blocked.** The earlier
>   STOP (b51469c: "4-12× SLOWER, 4T 0.24× / 1T 0.08×, x86-only") was an ARTIFACT of the naive `bf16_matmat_slice`
>   (scalar decode + `acc[64]` in L1 NOT registers, load/store every k) — exactly the condition it flagged. FIX
>   (2026-06-08): rewrote it as compile-time-B specializations `bf16_matmat_b1..b8,b16` — BV accumulators register-
>   resident (unrolled `for(j<BV)` → named scalars), rows blocked 2 at a time, broadcast-FMA auto-vectorized per ISA
>   under -march=native; naive intrinsic loop kept ONLY as `bf16_matmat_generic` fallback. Self-test still PASS
>   (matmat(B=8) vs B×matvec L2_rel ~3e-7); golden untouched (opt-in). **Re-bench 0.6B M1 K=50: 4 threads 0.24×→2.10×;
>   1 thread 0.08×→0.88×.** B-sweep @4T: B=2 1.87×, B=3 2.01×, B=4 2.23× (sweet spot), B=8 2.21× (B=6/16 dip ~1.6× =
>   noise/reg-pressure). Why it flips: single-stream re-reads weights B× → saturates the shared mem controller across 4
>   cores (bandwidth-bound); batched reads weights ONCE. ~2× = the bf16 ceiling (premise microbench was right); int4/int8
>   should push past it, x86 (bandwidth-bound) ≥ this. **DECISION REVERSED: the `--batch` integration IS worth building
>   on M1** — it's the AUDIOBOOK/long-text lever (split a paragraph into 2–4 chunks, step batched → ~2× wall-clock vs
>   sequential, reusing weights). **(2) int8/int4 batched twins DONE (2026-06-08, commit 720e939):** `qwen_matmat_int8`
>   (int8 W + f32 act) + `qwen_matmat_q4_0` (nibble unpack amortized over B), self-test correctness PASS, `make
>   matmat-bench` (`--matmat-bench`) times REAL kernels. M1 B=8: **int4+batching WINS at 1T AND 4T (1.1–1.4×) = THE M1
>   lever** (q4_0 single-stream is unpack-bound → batch unpacks once; int4 otherwise slowest single-stream → exactly
>   where batching earns its keep); bf16 wins 4T (1.4–1.8× kernel / 2.1× pipeline); **int8 loses/break-even because the
>   int8 SEQUENTIAL uses fast SDOT and the twin accumulates f32 (throws SDOT away) → TODO: integer-dot int8 twin
>   (SDOT ARM / VNNI x86)**. NEXT: (1) per-stream sampling + ragged-EOS + chunk scheduler (split text / keep batch full /
>   re-stitch via render_spans) → wire `--batch` (works at bf16 today, pair with int4); (1b) the int8-SDOT twin; (3)
>   validate on Ryzen/Turin. See docs/batching.md "CORRECTION (2026-06-08)" + "int8/int4 twins MEASURED".
> - **`--batch` MILESTONE A DONE (2026-06-08, commit after 720e939): long-form chunked synthesis CLI.** Sentence-aware
>   splitter (top-player practice: segment on `.!?;`/newline with decimal + abbreviation guards — "18.30"/"Sig."/"Ecc."
>   NOT split — then greedy-pack sentences up to `--batch-words` N (default 16, min N/3), merge sub-min trailing
>   fragment). `--batch` synthesizes each chunk via the EXISTING single-stream path + concatenates via `render_spans`
>   (seams land on sentence pauses → inaudible). `--batch-dry` previews the chunking without synth (tune the policy fast).
>   Milestone A = correct audio + the sequential baseline, and validates chunk-concat QUALITY before investing in batched
>   compute. Additive (normal/compose untouched, self-test PASS). **MILESTONE B (NEXT, the speedup): swap the inner
>   per-chunk loop for the batched compute kernels.** Hard parts: (1) RAGGED — each chunk's prompt prefills to a different
>   length → need per-sequence `kv_len_b` in `qwen_batch_talker_step` (the attention is ALREADY per-sequence; only the
>   shared `pos` → per-seq `pos_b` in the per-b RoPE/KV-append/attention loops; batched matvecs are position-agnostic →
>   unchanged); (2) prefill each chunk's prompt into bb's per-seq KV; (3) per-stream sampling (own RNG/rep-penalty per
>   chunk) + ragged EOS (drop finished chunks, compact batch); (4) feed sampled codes through batched CP + per-chunk
>   decoder → concat. Pair `--batch` with int4 (the M1 lever). THEN the Promessi-Sposi single-vs-batched timing test.
> - **`--batch` MILESTONE B DONE (2026-06-08): batched compute wired in = the speedup.** B1 (commit after a6210d3):
>   `qwen_batch_talker_step_ragged` — per-seq positions (`pos_arr[b]`) + `active[b]` (NULL=lockstep, back-compat). B3:
>   `qwen_tts_generate_batch` (qwen_tts.c) — groups of ≤8 chunks; per group prefill each via the normal path (new gated
>   `ctx->prefill_only` early-return captures KV+seed hidden), then ragged batched gen (per-stream sample/EOS + batched
>   CP), then seam-free per-chunk decode + concat. `--batch` calls it for nc≥2, falls back to sequential on -2/error.
>   **WIRING PROVEN: `QWEN_BATCH_FORCE_MATVEC=1` (bit-exact proj) → mel_corr 1.00000 vs single-stream.** Real matmat =
>   valid alternative kernel (greedy fp-order trajectory fork, like int8; mel_corr ~0.68 vs single at temp0 — validate by
>   EAR). **M1 measured (0.6B bf16 temp0): 3-chunk paragraph 29.3s→19.4s, RTF 1.30→0.74 (sub-realtime); grows with chunk
>   count.** Additive (prefill_only gated; normal/compose/self-test unchanged). **NEXT: (1) int8/int4 in the batched step
>   (B2) — the model's mmap-resident bf16 weights make it work today but WITHOUT quant speed; wire qwen_matmat_int8/q4_0
>   into batch_proj using quantized weight fields → int4 is the M1 lever; (2) optional auto-activate `--batch` on long
>   text; (3) Promessi-Sposi single-vs-batched A/B across bf16/int8/int4.**
> - **`--batch` B2 DONE (2026-06-08): int8/int4 inside the batched step.** `qwen_batch_proj_q` (precision-aware batched
>   proj, dispatches q4>int8>bf16 via the matmat twins) wired into all Talker+CP projections; `cp_lm_argmax` fixes the
>   batched CP lm_head (was bf16-hardcoded → forked the trajectory in quant mode — the one real bug, found via
>   force_matvec). **WIRING PROVEN per precision: batched force_matvec == sequential split = mel_corr 1.00000 for bf16 AND
>   int8 AND int4.** self-test + golden (0.6B/1.7B, bf16+int8) still 1.0. **M1 clean isolated (0.6B, 7-sentence ~36s
>   audio, temp0): bf16 single 53.5s→batched 30.7s (RTF 1.48→0.78, 1.74× — the big win); int8 single 28.6s→batched 29.3s
>   (RTF 0.89→0.82, ~break-even — int8-single already SDOT-fast/sub-realtime, little headroom + the int8 twin is f32-accum
>   not SDOT); int4 slower on M1 (x86 lever).** Diagnostics: QWEN_BATCH_FORCE_MATVEC=1 (bit-exact proj), QWEN_BATCH_SEQ=1
>   (force sequential ref). **NEXT: (1) int8-SDOT batched twin (qwen_matmat_int8 integer-dot, the matmat-bench TODO) →
>   makes int8+batch win on M1 too; (2) batched CP lm_head still bf16 in the FORWARD-feed path? no — fixed; (3) Promessi-
>   Sposi A/B (have the tooling); (4) optional auto-activate --batch on long text; (5) validate batching on x86 (Ryzen/Turin).**
> - **TODO (later, this branch) — `--batch` × {server, streaming} interaction. EMPIRICAL STATUS CHECKED 2026-06-08:**
>   `--batch` + `--serve` → the server **ignores --batch** (serve runs its own serial single-stream request loop; batching
>   of CONCURRENT requests is NOT implemented). `--batch` + `--stream` → if the text splits to 1 chunk it falls back to
>   sequential (which DOES stream); if 2+ chunks the batched orchestrator decodes per-chunk **post-hoc** (not via the
>   streaming decoder thread) → **streaming is silently ignored, whole WAV written at end**. So today they DON'T compose.
>   Open questions to decide: (a) TTFA — batching prefills ALL chunks before generating → first-audio worse vs single-
>   stream chunk-1-first (measure); (b) is batched-streaming worth wiring (emit chunk-0's audio as soon as it EOSes while
>   the rest keep stepping)? (c) server-side request batching = the real throughput play for `--serve` (separate, bigger
>   feature — see next bullet). Likely doc outcome: **batching = throughput/long-form lever; streaming = latency lever —
>   pick per use-case.** Test via `bench_matrix.sh --full`. DESIGN DECISION (user 2026-06-08, agreed): **streaming stays
>   single-stream** (latency lever, one progressive sequence — nothing to batch within one request); batching belongs in
>   the SERVER as concurrent-request batching (below).
> - **TODO (BIG — the real server throughput feature) — SERVER REQUEST-BATCHING (continuous/dynamic batching).** User's
>   idea (2026-06-08) and the intended use of the batched kernels: N concurrent requests from DIFFERENT users (different
>   text/voice/params) stepped TOGETHER through Talker+CP (weight-stationary) → ~N× server throughput on bandwidth-bound
>   boxes (memory §B: 2.3–3.7× at N≥6–8). **The hard part is DONE**: `qwen_batch_talker_step_ragged` (per-seq pos) +
>   per-stream sampling/EOS + independent per-seq prefill are exactly this — `qwen_tts_generate_batch` is ~90% reusable
>   (the B sequences come from different requests instead of chunks of one text). **What's missing = the SCHEDULER +
>   concurrent server** (today `--serve` is serial single-thread): (1) concurrent accept + request queue; (2) batch
>   scheduler — start with **dynamic batching** (collect a ~10–50ms window or up to B, run to completion, respond all;
>   simple) then **continuous batching** (admit new requests into slots freed by EOS'd ones — compact+refill, vLLM-style,
>   max utilization); (3) per-request response routing (+ per-request SSE if streaming). Each request keeps its own
>   prompt/prefill (incl. its own `.qvoice`/speaker/temp) — bb already has per-seq KV. This is the CORRECT version of the
>   old `--workers` dead-end (that was N parallel GEMVs re-reading weights N×; this reads once). Pays most on x86 EPYC/
>   Sapphire — validate there. Likely its own branch off feat/batching.
>   **KEY INSIGHT (user 2026-06-08): continuous batching + STREAMING COMPOSE** (this resolves the "--batch+--stream don't
>   compose" finding, which was only about LONG-FORM post-hoc decode). Continuous batching steps the whole batch ONE frame
>   at a time → after each batched Talker+CP step every active request has a new frame → emit it to THAT request's stream
>   immediately (per-request SSE), decode incrementally per request. So: **batched Talker+CP (throughput, shared) + per-
>   request streaming decoder + SSE (latency) TOGETHER** — exactly how vLLM streams tokens to N concurrent users. TTFA is
>   GOOD (a new request is prefilled + admitted in-flight, starts producing frames at once — the "prefill all first" TTFA
>   worry was only the single-text long-form path). Cost: N per-request streaming decoder states (cheap vs batched Talker+
>   CP). This is THE production-serving architecture; continuous (not static) batching is the right target (static wastes
>   utilization on ragged EOS). Most complex feature of the batching arc → dedicated branch, validate on x86.
> - **TODO (later, this branch) — `--batch` × `.qvoice` + EMOTION mixing (user's "2 tests in 1", 2026-06-08).** Verify
>   batched long-form works with a loaded `.qvoice` (e.g. Silvio): the voice is a KV/WDELTA prefix — does each chunk's
>   cold-prefill (prev_prefill_len=0) correctly RE-APPLY the voice prefix in the batched orchestrator every time (no voice
>   drift across chunks)? Check fidelity (ear + mel-corr vs sequential `.qvoice`) + RTF, bf16/int8. **AND** test combining
>   `--batch` with `--emotion`/`--roughness` (the expressivity steering) — does the control-vector/steer apply per chunk
>   in the batched path, and can we mix emotions cleanly across a long text? (the batched step reuses the same Talker/CP;
>   steering is applied where? confirm it flows through). A/B: silvio_17b.qvoice on 1.7B + a 0.6B qvoice, single vs --batch,
>   plain + with emotion. 2-in-1: qvoice-correctness AND emotion-in-batch.
> - **TODO (perf epic) — REAL-HARDWARE validation campaign (needs rented/borrowed boxes; see docs/hardware-testing.md).**
>   The cross-CPU/SIMD optimizations (BFMMLA/SMMLA/SME on ARM, VNNI/BF16/AMX on x86) can only be validated off-M1. Run
>   `make bench-matrix` + `--caps` on each and paste results into docs/hardware-testing.md §5. Target boxes, in order of
>   access: (1) **user's Ryzen 6800H mini-PC** (Zen3, AVX2 — already on LAN, `tests/x86_bench.sh`); (2) **AMD server VPS**
>   (Scaleway EPYC Genoa/Turin = AVX-512+VNNI+BF16 — user has used Scaleway 9555P before); (3) **Mac mini M4** (rent/buy —
>   bf16+i8mm+SME, the biggest untested ARM lever); (4) **a powerful ARM with max SIMD we support** (AWS Graviton3/4 =
>   bf16+i8mm+SVE, or NVIDIA Grace = SVE2). For each: caps fires? self-test PASS (native+fallback)? RTF single/batch/
>   stream/server × bf16/int8/int4? Then fill in the §7 newer-ISA twin and re-measure. This is the standing perf loop:
>   **M1 dev → check-isa → rent → bench-matrix → fix → loop.**
> - **SPECULATIVE DECODING analysis (TODO, user 2026-06-07) — docs/speculative-decoding-analysis.md.** Model has an
>   INTRA-frame MTP (the Code Predictor = `small_to_mtp_projection`, 15 RVQ residual passes), NOT a next-frame
>   speculator. Ideas: (A) cross-model draft 0.6B→1.7B `code0` + batched verify; (B) training-free lookahead/Jacobi on
>   code0 (Medusa/Eagle OUT — need trained heads); (C) CP residual spec (risky, quality-sensitive); (D) spec-decode ⊂
>   batching (the parallel-verify IS a batched forward → build batching first). **DECISION NUMBER = 0.6B↔1.7B code0
>   acceptance rate** — cheap instrument-only experiment (reuse quant-ladder teacher-forcing rails) to run FIRST.
> - **MTP DEEP-DIVE (TODO, user 2026-06-08) — a dedicated analysis task, NOT this session.** Question to answer
>   precisely, with code citations: (1) **What MTP is in THIS model and how it works** — the weights contain a
>   `small_to_mtp_projection` + the Code Predictor (15 RVQ residual passes/frame). Is that a true Multi-Token-Prediction
>   head (predicts *future frames* à la DeepSeek-V3 / Qwen3 MTP) or only the INTRA-frame residual quantizer (predicts the
>   15 residual codebooks of the CURRENT frame)? Read the HF config / `MODEL.md` / the projection wiring to settle it —
>   current belief (line ~712) is INTRA-frame, NOT a next-frame speculator, but verify against the actual tensor graph.
>   (2) **Do we USE it today, and how?** Trace where `small_to_mtp_projection` feeds in `qwen_tts_code_predictor.c` /
>   `qwen_tts.c` — confirm it's the CP path we already run every frame (yes/no, with line refs), or dead weight we ignore.
>   (3) **Would it help SPECULATIVE DECODING / speed?** IF there is any next-frame-predictive capacity in the MTP head,
>   it could draft the next frame's code0 for a batched verify (spec-decode ⊂ batching — see the bullet above). IF it's
>   purely the intra-frame residual predictor, it does NOT give free next-frame drafts and the spec-decode lever stays
>   the cross-model 0.6B→1.7B code0 idea. Deliverable: `docs/mtp-analysis.md` + a verdict line in PLAN. Cheap, instrument-
>   /read-only; gate the spec-decode work on its conclusion. Don't conflate with the `feat/batching` matmat work.
> - **Quant:** int2 / int3 where we know it's viable (q2 already exists for the roughness path — extend as a
>   real quant tier?). Measure quality cliff per codebook (we have the quant-ladder instrument). NOTE: int4+batching
>   synergy (above) raises the value of a solid int4 path.
> - **Speed+quality benchmark of voice CLONE and preset voices** (systematic A/B: RTF + mel-corr + ear, per voice).
> - **Expressivity — EMOJI in the prompt:** what does the model do with 😄 😂 😢 🤔 / "lol"/"ahah"? Does it unlock
>   laughs/cries? (first quick test 2026-06-07: no error, tokenizer eats them as UTF-8 — needs ear check; samples/emoji/).
> - **Expressivity — SOUND-DISCOVERY SUITE:** a labeled grid (id → {text, lang, voice, steer/rate/vol}) that mass-
>   generates onomatopoeia/vowel/emoji combos PER LANGUAGE × PER PRESET VOICE; user listens + annotates the best by id;
>   bake the winners into a reusable paralinguistic library (macros). Goal: map which strings→sounds (laughs, cries,
>   gasps, sighs) each voice/language can produce. Faster than the ad-hoc one-off loop we've been doing.
>   - **VALIDATED SOUND BANK (ryan, ear 2026-06-07):** baked → [sigh]=Hah, [hmm]=Hmmm, [mmm]=嗯, [mah]=Mah,
>     [uhm]=Uhm, [hmpf]=Hmpf, [ahh]/[relief]=Haaa, [phew]=Uao, [laugh]=Eheh(IT), [haha]=Haha(EN), [heh]=Hehhh,
>     [ouch]=Ouch(EN)/[ahi]=Ahi(IT). EXTRA leads to bake: CN 哈哈/嘿嘿/呵呵 = 3 clean laughs; CN 唉 = weary sigh;
>     😂 = faint sigh; "Ahahah 😆" → AHI pain; ouch_ahi/laugh_hihi = sharp-pain "ahi" (reduce metallic). NOT good:
>     emoji 😢/lol, cry_buaa, sniff, boh, gasp_ah. RULES: macro rate ≥0.90 (metallic else); trailing "..." can add
>     a 2nd spurious vocalization; "metallic" = WSOLA over-stretch.
> - **Known model quirk (note):** occasionally elongates the letters of short words intermittently (model-side,
>   not the sad recipe) — "Che fatica" drew out even at neutral rate 1.0. Investigate if it becomes a problem.
> Uncommitted at stop: none of the repo (all committed: 52a1cf8→62da9ed→7b60e75→0c8ce04). Local-only: `voices/galatea_06b.qvoice`,
> `samples/emergent/` (sbuffo audio), `/tmp/qwen3tts_src` (official source, ephemeral). Two stray `analisi_leak*.md` left untracked (not ours — decide later).

> **Status 2026-06-05:** built + ear-validated on branch `feat/expressivity` (6 commits, NOT merged).
> Two levers ship; the feature WORKS and is a real differentiator, but it is a **v1, not the ceiling** —
> the control-vector method is crude additive steering and several gaps remain (below). Mechanism
> (confirmed): RVQ split — Talker codebook 0 = words (intact), CP codebooks 1-15 = fine texture/prosody;
> perturbing the CP changes delivery without losing intelligibility. `--instruct` alone barely moves
> delivery (known Qwen3-TTS limit, even full-size on GPU) — these levers amplify it.

**SHIPPED (works):**
- **`--roughness <0..1>`** — per-frame blend of a q2 copy of CP `down` into the high-prec output
  (`down` = causal driver of the q2 "death-metal" texture). Continuous grit/anger knob, ear-validated TOP.
- **`--emotion <name>`** — calibrated control-vector presets (`presets/emotions/` EN + `it/`): 9 tones
  (happy/excited/eager/proud/sad/gloomy/news/dramatic/calm), weight baked in, **blendable**
  (`happy:0.5,proud:0.5`), global `--steer-weight` dial. Capture via `QWEN_STEER_CAPTURE` + `tests/steer_make.py`
  (diff mean(instruct)−mean(neutral) at the single Talker→CP injection point `code_predictor.c:731`).
  **Cross-model** (CP identical 0.6B/1.7B). Default-off → golden mel-corr 1.0.
- **Voice-clone auto-trim** (`qwen_trim_trailing_silence`) — drops a faded ref tail so clones don't learn a decrescendo.

**WHY NOT THE TOP YET (open gaps — the honest list):**
- [ ] **`happy` is a per-voice TIMBRE limit.** Transfers fine on ryan, but on a soft narrator clone (Silvio)
  it fades at every weight — *even when captured natively on that voice* — because the voice has no bright
  register to reach. No additive direction creates a register the voice lacks. → some tones are intrinsically
  voice-bounded; need to detect/flag this, or a non-additive method (pitch/energy contour control?).
- [ ] **Directions are voice-specific & over-steer differently per voice** (Silvio excited TOP@0.7, voice-drift@1.4;
  ryan happy needs 1.4). No systematic ryan↔voice calibration map yet — today it's hand-tuned per voice
  (`presets/emotions/<voice>/`). Want an auto-calibration pass (sweep + pick the clean knee per voice).
- [ ] **Steering is crude additive `cp_x += w·vec` at ONE point.** Likely better: per-CP-layer injection,
  normalization to stay on-manifold (over-steer → metallic), contrastive/mean-centered directions captured
  over MANY sentences+speakers (current = 1 voice, ~100-150 frames), or steering the late codebooks (c11-15 =
  the measured texture surface) specifically.
- [ ] **Only ~9 tones mined, EN+IT only.** The "weight = mood crossfade" trick (one capture → 2-3 moods at
  different weights) is barely scratched; whisper/sarcastic/confidential/ironic/authoritative-variants untouched.
- [ ] **Mood-blending** (`a:0.5,b:0.5`) implemented but barely explored — could give a continuous 2-D mood space.
- [ ] **No segmenter/markup yet** (`[pause]`/`[emph]`/`[voice:]`/`[instruct:]`/per-span emotion) — the original
  long-form/audiobook vision. Per-span emotion switching is the natural next product step.
- [ ] **Not in the test suite** (no `--emotion` smoke / no steering golden); not merged to feat/labs.
- **Per-layer & cross-speaker/language sensitivity still largely unmeasured** (validated 0.6B+1.7B / ryan+Silvio / EN+IT).

**NEW LEADS (2026-06-06) — palette decorrelation + prosodic axes:**
- [x] **ROOT CAUSE of "all tones sound alike" FOUND & FIXED (no re-capture).** Diagnosed the shipped IT
  palette: vectors are highly collinear — mean off-diagonal cosine **+0.57**, and the somber cluster
  (gloomy/news/dramatic/calm/sad) is +0.75..+0.91 (≈ the SAME vector). Each preset is ~80-90% a shared
  "I-am-being-instructed-vs-neutral" common mode; the emotion-specific part is only the residual (happy 86%,
  but gloomy/news/dramatic only 46-55%). Fix = **mean-center the palette** then renormalize:
  `vec' = β·mean_all + γ·(vec − mean_all)`, renorm to original per-emotion norm. β=0 γ=2 →
  **mean pairwise cosine +0.57 → −0.09** (near-orthogonal), same magnitudes. Tooling: `tests/steer_center.py`.
  → bake this as the shipped palette (or a `--decorrelate` build step); also re-capture over MANY
  sentences+speakers so the common mode is weaker at the source. Directly closes the "crude additive / not
  contrastive" gap above. STILL TODO: ear-validate the centered palette is more distinct AND still natural
  (centering can push off-manifold → metallic); pick β/γ knee; rebuild EN + IT + per-voice.
- [ ] **Galatea is the better IT base voice** than Silvio (CC/PD LibriVox `it_galatea_fasol.wav`, Riccardo
  Fasol). Neutral RMS 70 vs Silvio 29 → full/clear vs soft/low; emotions land better on it. New
  `voices/galatea_06b.qvoice` (823MB, trim only 0.03s = no fade tail). Use as the reference IT clone for
  expressivity demos. (1.7B + native-Galatea palette = open.)
- [ ] **Explicit prosodic axes: `--volume`, `--rate`/`--speed`, `--pitch` (user req 2026-06-06).** Same
  control-vector machinery as `--emotion` — capture "speak louder/softer", "faster/slower", "higher/lower"
  as instruct-vs-neutral directions on `cp_x` and expose as signed knobs (−1..+1). (This is WHY some emotions
  already shift Silvio's volume — the captured direction carries projection/energy.) Two implementations to
  compare: (a) **steering vector** (model-side, changes delivery/register, on-manifold risk) vs (b) **cheap
  DSP post-step** — `--volume` = PCM gain/normalize (trivial, `qwen_tts_audio.c`); `--rate` = WSOLA/`atempo`
  time-stretch (pitch-preserving); `--pitch` = pitch-shift. DSP is reliable & decoupled but doesn't change
  *delivery*; steering changes delivery but can distort. Likely ship: DSP for volume/rate (predictable),
  steering-vector for "projection/intensity" (the expressive one). Volume/rate are the easy wins — do first.
- [x] **MEASURED 2026-06-06 (Galatea 0.6B, neutral text, ear-validated):** the per-mood recipes that actually work:
  - **Steering saturates fast & asymmetrically.** Default `--steer-weight 1.0` is too timid; on the CENTERED
    palette w≈1.8–2.6 roughly doubles movement and stays clean (no clipping, hi6k flat → on-manifold) for the
    UP moods. The DOWN moods (sad/gloomy/calm) DON'T respond to more weight — w4.2 is ~= w2.6, gloomy even goes
    MORE monotone (F0std 17→10). So weight is an UP-mood lever only.
  - **Down/somber moods = rate↓ + volume↓, NOT steering.** "Sadness" in speech is tempo+energy, which a
    single-point CP injection can't impose. DSP slowdown (`atempo 0.82–0.85`, pitch-preserving) makes sad/gloomy
    read as genuinely downcast — ear-confirmed "credibile". This is why `--rate`/`--volume` are REQUIRED, not extra.
  - **Joy = excited (NOT happy) + rate↑ + volume↑.** The `happy` direction loses energy when pushed (the
    "neutral is already upbeat" ceiling). The `excited` direction pushed (w2.6) + `atempo 1.10` + `volume 1.10`
    = RMS +34%, F0 +13Hz, faster → genuine joy. Ear-confirmed. Lesson: for bright/joy use excited, not happy.
  - **`--roughness` is TIMBRE (raspy/worn/smoker voice), NOT rage.** Ear-confirmed: roughness on a clean qvoice
    just makes it gravelly/phlegmy, not angry. Rage = a CAPTURED angry steering direction + volume↑ + rate↑
    (sharp/tense prosody), with NO roughness. Keep `--roughness` as a voice-CHARACTER knob, not an emotion.
    (Capturing angry directions on 1.7B in progress.)
  - **Design consequence — preset MANIFEST.** Real emotions are COMPOUND = `{vec, steer_weight, rate, volume,
    roughness?}`, not a bare vector. Extend the `.vec` palette to a manifest (name → bundle) so `--emotion joy`
    / `--emotion rage` sets all knobs at once. Bake centered-palette + per-mood weight + rate/volume here.
  - **FULL RAGE is OUT OF REACH (model limit, ear-confirmed).** Captured 3 "furious/hostile/yell" directions on
    1.7B, centered them — they land on **proud/authoritative/emphatic**, not angry (the model converts anger into
    forceful control). Adding low roughness (0.25–0.40) + agitation (atempo 1.12) gets to **"annoyed/stern/light
    irritation"** ("tired prof telling the class: hey, meeting tomorrow!") — useful and credible, but NOT furious
    rage. Real anger needs tense/strained phonation + irregular bursts that additive single-point CP steering
    can't impose. → ship as `annoyed`/`stern`, document rage as a known hard case. Artifacts persisted in
    `presets/emotions/it_centered/` (decorrelated palette + `angry.vec` + README with the recipe table).

- [ ] **VoiceDesign expressivity — INVESTIGATE (Leo Yu 2026-06-06).** Leo reports CustomVoice/clone "not as
  emotional/expressive even with good instruct prompting" while **VoiceDesign is very expressive but can't lock a
  voice identity** — the exact CustomVoice-vs-VoiceDesign tradeoff our steering work bridges. TASK: downloaded
  `qwen3-tts-voice-design` (1.7B, 4.2GB); generate angry/sad/happy/excited/fearful with RICH official-style prompts
  (multi-clause, e.g. "incredulous tone with a hint of panic creeping in", "vowels still tighten when nervous" —
  the official examples are descriptive, NOT bare "speak angrily"), fixed voice-description + varied emotion clause,
  EN+IT. LISTEN. IF VoiceDesign really renders rage/sadness better → reverse-engineer WHICH levers it has that
  CustomVoice lacks: hypothesis = (a) no locked speaker identity → full prosody freedom (the identity-conditioning
  that pins CustomVoice delivery is absent); (b) trained on rich descriptive style prompts. Levers we might port:
  feed richer instruct, or relax speaker conditioning strength during steering. Official emotion control IS in
  VoiceDesign per the paper (SOTA on InstructTTSEval); the limit is it doesn't clone.
  **TESTED 2026-06-06 (downloaded VD 1.7B, EN+IT emotion matrix, user ear-verdict):**
  - VoiceDesign DOES make bigger acoustic excursions than our CustomVoice (IT: angry F0 **+92Hz** & F0std +150%,
    happy +29Hz, sad slowed +95%) — confirms the identity-clamp hypothesis: our locked speaker embedding/WDELTA
    pins the register and blocks the pitch swings strong emotion needs. It also CORROBORATES our recipes (VD does
    sad by slowing hugely, joy by raising pitch+rate — same directions, just more extreme).
  - **BUT user verdict = VD's superiority is largely a "different-speaker ILLUSION":** VD synthesizes a NEW voice
    per prompt, so the listener reads emotion easily because they're literally different people. On ONE consistent
    recognizable voice the same shift is HARDER to perceive — no "clean break" (sighs/laughs/pauses) a human expects.
    Our task (emotion on a STABLE identity) is intrinsically harder AND more valuable. Also: **VD Italian is WORSE
    than ours** ("più orecchiabile noi"); angry/excited rendered with poor Italian — VD has NO base-speaker param
    (voice is 100% prompt-derived → lands on a base weak in IT, likely a Chinese one), whereas we always use ryan
    (strong in IT/ES/FR). VD also follows the target language inconsistently. → OUR consistent-voice direction wins.
  - **Engine note:** `--load-voice` CAN inject an embedding INTO VoiceDesign (cross-model, qwen_tts.c:703, cosine
    ~0.94) → could pin our identity + use VD's expressive instruct; likely re-clamps, but a cheap experiment.
  - **NEXT LEADS (readable emotion on a consistent voice):** (A) **relax-identity lever** — scale down speaker
    conditioning (embedding/WDELTA strength) during emotional spans to unlock pitch excursions while staying
    recognizably the same voice (THE key reusable engine experiment). (B) **strong contour + onset "stacco"** —
    sad span STARTS slow+quiet+falling, angry STARTS loud+fast+high, with a short PAUSE before the span = the
    clean break audiobook voice-actors use. (C) pauses/tempo/volume swings we already have; sighs/laughs we don't
    (model limit). Fold into the compound-emotion manifest.

- [x] **Hidden-tag / special-token hunt — DEFINITIVE NEGATIVE (2026-06-06, first-hand model analysis).** Goal: find
  special chars/tokens to pilot the model (sighs/laughs/onomatopoeia/emotion). Result: **Qwen3-TTS has NO
  paralinguistic or emotion tokens.** Full `added_tokens_decoder` (both 1.7B CustomVoice + VoiceDesign) = only
  standard Qwen tokens (im_start/end, vision, tool_call, fim, think) + TTS framing (`<|audio_start/end|>`,
  `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<|audio_pad|>`). NO `<laugh>`/`<sigh>`/`<breath>`/emotion tags.
  Corroborated: (a) GitHub Discussion #238 is an OPEN request to ADD inline emotion tags (so they don't exist);
  (b) third-party "TTS-Audio-Suite" gets laughs/sighs NOT from Qwen but by post-processing through a SEPARATE model
  **Step Audio EditX** (inline tags `<Laughter:2>`/`<emotion:happy>`/`<style:whisper>`; covers ZH/EN/yue/ja/ko only,
  "distorts" other langs → not for Italian). **The `codec_think_id/nothink/think_bos/think_eos` (2154-2157) = the
  LANGUAGE-conditioning slot** (`qwen_tts.c:947`, `[THINK,THINK_BOS,language_id,THINK_EOS]` with `-l`, else
  `[NO_THINK,...]`), already used by our engine — NOT a free-form instruction-following reasoning trace we're
  missing. **CONCLUSION: emotion control in Qwen3-TTS = the instruct natural-language text ONLY** (no magic tags);
  real paralinguistics need a 2nd-stage audio-edit model (Step Audio EditX), language-limited. → our steering+recipes
  + (optional, EN-only) a paralinguistic post-editor is the only path to true sighs/laughs. Style-keyword inspiration
  from EditX for VoiceDesign instructs: whisper/news/radio/story/shout/warm/gentle/authority/serious/murmur/etc.
  Still worth a quick check: does OFFICIAL inference place the instruct differently than our `<|im_start|>user\n{instruct}<|im_end|>` (qwen_tts.c:882)?
- [x] **Relax-identity lever — NEGATIVE (2026-06-06, ear-confirmed).** Added `QWEN_SPK_SCALE` env knob
  (qwen_tts.c:~1100, default 1.0, scales the speaker-embedding contribution). Scaling DOWN (0.6/0.3) does NOT
  free pitch excursions — it removes drive (RMS 76→52, F0 range shrinks), and at 0.3 the voice becomes a
  DIFFERENT voice (lost Galatea); others not angrier. The register clamp lives in the WDELTA **weights**, not the
  embedding → embedding-scale misses it. Faithful-clone vs VoiceDesign-pitch-range = physical conflict (same clamp).
  Knob kept as a diagnostic. Deeper WDELTA-blend-toward-base lever = the VoiceDesign tradeoff we want to avoid.
- [x] **Instruct-by-language test (1.7B ryan, n=1, ear-corrected):** instruct moves CHINESE delivery hugely
  (angry ΔF0 +45Hz, ear-confirmed angry) but IT/EN by ±1-2Hz → **instruct/emotion is Chinese-tuned, weak on EU**
  (validates the user's hypothesis; justifies our CP-steering path for EU). CORRECTION: my acoustic proxy claimed
  "ZH instruct boosts Italian more" (ΔF0 -49) but BY EAR the EN-instruct IT was angrier and ZH-instruct IT less →
  the "Chinese instruct on EU text" boost is NOT confirmed. (Lesson: ΔF0 magnitude ≠ perceived anger; ear is truth.)
  Side-obs: `it_neutral` (no instruct) drifted to Chinese-ish PRONUNCIATION while `it_angry` was perfect Italian —
  possible language re-anchoring by instruct/think → folded into the think re-audit task.
- [x] **DEEP RE-AUDIT (task #6) — DONE, validated against OFFICIAL SOURCE (cloned QwenLM/Qwen3-TTS, not our code).
  Our RE was CORRECT; no hidden lever. Hypothesis REFUTED.**
  - `modeling_qwen3_tts.py:2135-2147`: think is a FIXED prefix — `[codec_think_id, think_bos, language_id, think_eos]`
    with a language, `[codec_nothink_id, think_bos, think_eos]` without. NOT autoregressive reasoning. EXACTLY our
    qwen_tts.c:947-955. The paper's "probabilistically activated thinking pattern" is a TRAINING thing; inference uses
    this minimal fixed slot. No `enable_thinking`, no on/off/levels, no generative CoT we skipped.
  - `modeling_qwen3_tts.py:2075-2080` + wrapper `qwen3_tts_model.py:269-276`: instruct is prepended as text embeddings,
    format `<|im_start|>user\n{instruct}<|im_end|>\n` then assistant text — IDENTICAL to ours (qwen_tts.c:882). Placement correct.
  - **Why instruct is weak (externally validated, NOT our bug):** (a) the official voice-CLONE API has NO instruct param
    (clone = ref_audio/ref_text only) → our `--load-voice + --instruct` is OFF-DISTRIBUTION; community confirms "instruct
    on cloned voices does nothing" (Qwen Disc #218/#231/#238, mlx-audio #453, HF #38). (b) instruct is Chinese-tuned, weak
    on EU even on preset voices (our test: ZH +45Hz vs IT +1.5Hz). (c) 1.7B > 0.6B for emotion control.
  - **CONCLUSION: the model genuinely offers no more via instruct for cloned EU voices → CP-steering (our control-vectors)
    is the ONLY path for emotion on a cloned Italian voice. All alt levers (think / hidden tags / relax-identity / ref-breath
    sighs) are closed with proof. Ship what works: centered palette + tempo + pauses.** Official src at /tmp/qwen3tts_src (ephemeral).

- [x] **Cross-LANGUAGE steering validation (2026-06-06, ear-confirmed, ES/FR/DE/JA/KO, centered palette on preset voices).**
  The IT/EN-captured steering does NOT transfer uniformly across languages (refines the old "presets are cross-lingual"
  belief → cross-MODEL yes, cross-LANGUAGE only partial): **JA = best** (angry/sad/happy all credible), **DE = weak**
  (emotions barely differ by ear — my acoustic proxy OVERSTATED it), **ES = broken** (angry sounds SAD, happy sounds
  aroused/breathy not happy), **FR = happy→sultry/breathy not happy**, **KO = sad+happy don't work**. **`happy` is the
  most language-FRAGILE direction** — in several langs it LOWERS pitch or induces breathiness that reads as "sultry"
  (consistent with happy being intrinsically weak + breathiness=happy correlate; `excited` is our real joy lever).
  → **emotion palette needs PER-LANGUAGE calibration** (or at least re-capture ES/FR/KO); and prefer `excited` over
  `happy` for joy cross-language (untested — next quick experiment). Voices used: ryan (ES/FR/DE), ono_anna (JA), sohee (KO).
  Lesson re-confirmed: acoustic ΔF0 proxy ≠ perceived emotion (proxy said DE clean, ear said weak) — ear is truth.
- [ ] **`--volume` knob (trivial, not yet exposed):** no `--volume` flag exists; volume is pure PCM gain (qwen_tts_audio.c
  one-liner) or ffmpeg post — unlike pitch (identity-clamped), volume has no obstacle. Fold into the compound-emotion manifest.

See [[project_expressivity]] for the full build log, the cross-voice map, and the Silvio re-clone saga.

### A2. Cross-CPU PERF — Leo Yu 9950X3D (Zen5 V-Cache) report, 2026-06-06 — VERIFIED LEADS
> Leo benchmarked pure-C v0.9.0 on a Ryzen 9 9950X3D (CCD0=96MB V-Cache, CCD1=32MB) across Docker/WSL2 +
> native Windows (first MinGW build). Deep-analyzed against our code. The genuinely-useful, verified takeaways:
- **int4×CCD insight (CORRECT, matches us):** int4 LOSES on CCD0 (60MB int8 fits the 96MB V-Cache → no bandwidth
  pressure → nibble-dequant is pure cost) and WINS on CCD1 (32MB<60MB → bandwidth-bound → smaller footprint pays).
  Refines our "int4 = the x86 lever, not M1" → precisely: int4 helps ONLY when the working set does NOT fit cache.
- **Thread oversubscription is catastrophic:** -j32 (all 16 cores/32 threads) → RTF 6.38, 3.9× WORSE than -j1
  (cache thrash + cross-CCD latency on a bandwidth-bound load). Confirms our 4-thread-sweet-spot / EPYC -j1>-j4.
  → PRODUCT IDEA: auto-detect X3D, pin CP to the V-Cache CCD, cap threads low. Never oversubscribe.
- **SD is BLAS-bound (verified: `cblas_sgemm` in speech_decoder.c).** Native Windows SD drain = 153 ms/f vs Linux
  20 (OpenBLAS Win32 CreateThread/WaitForSingleObject per-GEMM sync). This single number kills native Windows
  (RTF 2.10). **CHEAP FIX to try before any dual-boot: `OPENBLAS_NUM_THREADS=1` for the SD on Windows** — if the
  cost is per-GEMM thread sync, single-thread (no spawn/join) likely beats 153 → ~30-40, putting Windows-native at
  ~1.0 with NO Linux/hybrid. Or a persistent BLAS pool / our own Win32 pool for SD.
- **#1 UNTAPPED WIN — int8 the TALKER.** Leo's Talker is pinned at 21 ms/f across bf16/int8/int4 → tell that his
  Talker is NOT being quantized (stays bf16=1.2GB). On our tree `--int8` DOES quantize the 0.6B Talker (the
  `hidden<2048` gate was removed). Quantizing it → 600MB halves DRAM traffic on the bottleneck HE correctly named
  (Talker=DRAM-bound). Verify dtype via `--debug`; OS-independent win, ~−0.7..0.9s.
- **Docker CP=54 likely contaminated by 9P filesystem** (his own note: model on /mnt/c = +30% CP). Clean on ext4
  ≈ 38-42 → the "Windows 1.82× faster CP" (54→30) shrinks to ~1.3× and is partly mmap/filesystem, not V-Cache.
- **RTF/per-frame table don't reconcile:** int4-j4-CCD1 (CP60+Talker21=81ms/f) shows RTF 1.11 < int8-j1-CCD0
  (CP54=75ms/f) RTF 1.21 → only explained by prefill (threaded GEMM, faster at -j4) and SD-overlap differences;
  per-frame table ≠ whole-utterance RTF. The hybrid streaming RTF 0.91 ignores per-frame cross-VM IPC (optimistic);
  Linux-BM 0.80-0.84 stacks best-cases (Talker won't improve — DRAM-bound; CP 30→22 is speculative) → realistically
  0.85-1.0. The dual-boot is likely UNNECESSARY (fix Windows SD threading + int8 Talker instead).
- **Confirmed-good & worth keeping:** Leo's native Windows port (5 changes: CreateFileMapping mmap shim, O_BINARY,
  Win32 server stub, posix_memalign→aligned_alloc, OS detect) closes our "Windows won't compile" gap → ask for the
  patch. The `__AVX512VNNI__` Makefile-clean-bug he hit = our known gotcha. Cooperlake-BLAS-beats-AOCL on Zen5 SD
  (medium ConvNet shapes favor Ice Lake blocking; Zen5 has full AVX-512) — plausible, use `OPENBLAS_CORETYPE=COOPERLAKE`.
- **Streaming clarification:** Leo's "streaming not yet implemented" = HIS hybrid Win-CP→Linux-SD per-frame token
  PIPE only (`--emit-tokens`/`--decode-tokens`, his local patch, NOT in our tree). It does NOT mean we lack
  streaming — we HAVE `--stream` (TTFA, feat/streaming-ttfa) + OpenAI-compatible server streaming. Different layer.

### B. Weight-stationary batching = the throughput lever for the SERVER
> **Branch plan (2026-06-04):** this becomes its OWN branch (e.g. `feat/batched-generation` off
> `feat/labs`) — batch-dim in Talker/CP step fns, batched int8/bf16 GEMM, per-stream attention+sampling,
> continuous-batch scheduler. The perf bet for LONG-FORM/server (N≥6-8). NEXT cheap step before building:
> re-run the batching microbench with the REAL int8/bf16 kernels (not f32 BLAS) for the true threshold.
Single-stream (one audio) cannot reorder to keep weights cache-hot: the CP's 16 steps (and the
Talker's tokens) are a hard autoregressive chain (step g input = step g-1 output), so the weight
read order `L1..L5 x16` can't be reordered ("weights-outer, steps-inner" needs all step inputs at
once — they don't exist yet). BUT across **independent concurrent requests** the reorder is valid:
batch N requests -> each weight read is reused across all N (weight-stationary across the batch) ->
~Nx throughput. This is standard LLM continuous-batching. **Not a single-file latency win; a big
multi-request server throughput win.** Re-analyze when optimizing the HTTP server for concurrency.

**2026-06-04 — MEASURED: `--workers` (N parallel GEMVs) is NOT batching, and is a DEAD-END on M1.**
The `--workers N` pool runs N INDEPENDENT GEMVs, each re-reading the weights → on a bandwidth-bound
bf16 workload that's N× the DRAM demand → contention, not reuse. Bench (3-sentence text, full vs 3
concurrent reqs on `--workers 3`, 0.6B M1): per-chunk latency TRIPLED (~6s→~17s), aggregate throughput
+8% only, total wall-clock WORSE (17.7 vs 16.4s). A single 4-thread synthesis already saturates M1
bandwidth. **So the worker-pool is correctness-validated but NOT a throughput win on M1.** See
[[project_split_parallel_bench]] + `docs/pipeline.md`. The fix is TRUE batching (the N current-frame
vectors stacked into ONE GEMM `Y=W·X`, weight read ONCE) — a different code path from the worker pool
(needs a batch-dim in the step functions; the matvecs become GEMMs like prefill already does, attention
stays per-stream since each request has its own KV — but attention is <1% so the 90% matvec slice
batches cleanly). Microbench of the GEMV-loop vs GEMM ceiling: see section-B addendum below / `docs`.
NUANCE: after int8+SDOT the CP shifts toward compute-bound (PLAN "Speed must come from compute
throughput"), so the batching win is largest in the bf16 path and smaller (but still real, via better
GEMM FMA/SIMD utilization) in int8. **Decision: park worker-parallel behind real batched-GEMM OR a
higher-bandwidth box (Ryzen/server). The DUAL-PURPOSE insight — text-splitting is ALSO the unit of
prosodic control (track A) — means the EXPRESSIVITY half of splitting is the live track; it needs no
parallelism and stands alone. First-audio latency → sequential chunk-1-first (~6s vs ~16s), not
concurrency (which tripled chunk-1's latency).**

**2026-06-04 — MICROBENCH: true batched-GEMM DOES reuse the weights, but only pays off at N≥4-6.**
`/tmp/batch_bench.c` (Accelerate f32, M1): N sequential GEMV (= workers) vs 1 GEMM width-N (= batch),
on the real CP/Talker matvecs. speedup = gemv_loop / gemm:

| W | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|
| CP gate_up [3072×1024] | 0.77× | 1.20× | 1.66× | 2.31× | 3.23× |
| CP down [1024×3072] | 0.86× | 1.16× | 1.63× | 2.30× | 3.12× |
| CP QKV [2048×1024] | 0.53× | 0.84× | 1.17× | 1.77× | 2.46× |
| Talker-1.7B gate_up [6144×2048] | 0.80× | 1.15× | 1.66× | 2.69× | 3.74× |

- **Weights ARE reused** (the win mechanism is real): GEMM time is ~FLAT in N (gate_up 0.48ms at
  N=2..8) while the GEMV loop grows linearly — the weight is read once, extra columns ~free.
- **BUT the GEMM has a fixed cost ~2.5× a single GEMV** → N=2 LOSES, break-even N≈3, strong (2-3.7×)
  only at N≥6. So batching is a **server-scale / many-chunk lever, NOT a 2-3-sentence win** — exactly
  what this section always said ("not a single-file latency win").
- **Consequence for the use cases:** short one-shot (2-3 sentences, N≤3) → batching does NOT help on M1
  (neither do workers). LONG-FORM / AUDIOBOOK is the natural home: split a book into many chunks → keep
  N≥6-8 active in a continuous batch → **2.3-3.7× throughput**, the real lever. Server with many
  concurrent users → same regime.
- **Build cost (real):** batched generation needs a batch-dim in the Talker/CP step functions (matvecs
  → GEMM like prefill already does; attention stays PER-STREAM since each request has its own KV, but
  attention is <1%), per-stream sampling, and a continuous-batch scheduler. Caveat: this f32 Accelerate
  ceiling is the bf16-regime story; the int8+SDOT hot path is more compute-bound, so a batched int8
  kernel's win is smaller (still real via better FMA/SIMD utilization). NEXT before building: re-run
  this microbench with the ACTUAL int8/bf16 kernels (not f32 BLAS) to get the realized threshold/curve.

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
words, user ear-approved, int8 golden regenerated) · **x86 AVX2 + cross-OS pthread pool + int4 —
✅ VALIDATED on Ryzen 6800H 2026-06-04 (RTF ~2.02, int4/4-thread; AVX2 +6% over scalar, bandwidth-
bound)** · **x86 AVX-512/VNNI — written (d67648a/b89f30e), UNVALIDATED, needs a rented Zen4+/Intel
box (6800H is AVX2-only)** · server continuous-batching (throughput only).
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
