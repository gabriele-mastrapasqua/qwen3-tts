# Arm Linux parity for the v2 serving generation and C12-WIN — deferred track

**Task.** Record, with an explicit verified/unverified split, what the v2 serving generation
and the C12-WIN work leave broken or unreached on Arm Linux, so the work can start cleanly
once the VNNI/Turin track is closed. **Question.** Which of the reported items are facts
about the current tree, which are facts about a different tree, and which are opinions?
**Known facts.** Listed in section 2, each with the check that established it.
**Unknowns.** Everything in section 3, including every Arm serving number.

## 0. Provenance and how much weight to give this

The source material is an external review produced by another agent working from a **clone
of the public branch at commit `edfd3fb`**, not from this working tree. That matters:

* `edfd3fb` is behind this tree. The review states "admission slicing (C12-WIN-10):
  specification only, no code at this HEAD" — that is now stale, the state machine landed in
  `c10b547` (default off).
* Its Arm serving numbers come from a **heterogeneous 20-core box** (two core classes ~39 %
  apart, interleaved) at **concurrency 2 against a 2-slot server**, n = 12 per arm, one
  unpaired wave per arm, arms in separate processes. The review says so itself and calls it
  a structural probe rather than a capacity measurement. That self-assessment is correct and
  should be preserved: **no Arm capacity or lane verdict may be quoted from it.**

So: the structural and source-level findings are worth acting on, several are confirmed
below against THIS tree, and the performance conclusions are not evidence.

## 1. HIGHEST-VALUE FINDING, and it does not belong in this track

**The tree does not link when neither `__ARM_FEATURE_DOTPROD` nor `__AVX512VNNI__` is
defined.** Verified on this tree at HEAD, not merely inferred:

```
make clean && make blas ARCH_FLAGS="-march=armv8-a"
Undefined symbols for architecture arm64:
  _qwen_conv1d_int8_v2      _qwen_conv1d_int8_v2_available   _qwen_conv1d_int8_v2_cp
  _qwen_conv1d_int8_v2_ctx  _qwen_conv1d_int8_v2_pack
  _qwen_convt_pack_stack    _qwen_convt_stack_epilogue
```

Cause: the definitions live inside `#if defined(__ARM_FEATURE_DOTPROD) || defined(__AVX512VNNI__)`
(`qwen_tts_kernels.c` 9900-10432) while the declarations in `qwen_tts_kernels.h` and the
call sites in `qwen_tts_dispatch.c` / `qwen_tts_speech_decoder.c` are unconditional. The
`#else` fallback at :10228 sits INSIDE the outer guard, so it is unreachable when that guard
fails, and it does not cover the ConvT pair at all.

**This is not an Arm-parity item and should not wait for this track.** It breaks
`SIMD=portable` — the default non-VNNI x86 target — and `SIMD=scalar`, on every host. It is
also partly self-inflicted by the C12-WIN work: `_ctx` came with `ddfa5d8`, `_pack_stack`
and `_stack_epilogue` with `edfd3fb`, and the `_cp`/`_pack` pair with the earlier V2 packer.

Fix, small and mechanical: close the ISA block and follow it with an unconditional
`#if !defined(__ARM_FEATURE_DOTPROD) && !defined(__AVX512VNNI__)` section defining all
seven as the existing no-ops. Move `qwen_convt_pack_stack` / `qwen_convt_stack_epilogue`
out of the ISA guard entirely — they are f32 GEMM-packing and epilogue code with no
intrinsic in them, and were only ever inside it by placement accident. Then add
`SIMD=portable` and `SIMD=scalar` link-only jobs, because this break is invisible to anyone
developing on a VNNI or dotprod host.

## 2. Confirmed against THIS tree

| # | claim | check run here | result |
|---|---|---|---|
| 2.1 | link break without dotprod/VNNI | `make blas ARCH_FLAGS="-march=armv8-a"` | CONFIRMED, seven symbols, see section 1 |
| 2.2 | the five newest decoder flags are undocumented | `grep -c` in `docs/feature-flags.md` for `QWEN_SD_RES1_V2`, `QWEN_SD_GLUE`, `QWEN_SD_CONVT_STACK`, `QWEN_SD_BF16_PREUP`, `QWEN_SD_LANE_SPLIT` | CONFIRMED, **0 occurrences each** |
| 2.3 | the decoder lane has no ISA guard | read `qwen_lane_split_prepare` | CONFIRMED — `#if defined(__linux__)` only, no intrinsic |
| 2.4 | the handoff doc wrongly calls the lane x86-only | read `.work/turin-vnni-final-handoff-20260909.md` | CONFIRMED — **corrected in this commit** |
| 2.5 | `g_mm_gate[]` has no KleidiAI int8/bf16 rows | grep the table | CONFIRMED — only `QWEN_MMK_KLEIDI_Q4` has a row |

2.3 plus 2.4 is the substantive finding of the review: **the decoder lane, the mechanism of
record on the Turin product profile, ports to Arm Linux unchanged and no Arm profile sets
it** — and the reason it was never tried is most likely a wrong sentence in our own handoff.

## 3. Reported but NOT verified here

Recorded so nobody treats them as established. Each needs one command on an Arm Linux host.

* §1.2 the Arm product profile fails its own preflight because the KleidiAI probe falls
  through to the `int8 SMMLA` row while the real dispatcher calls KleidiAI first. The gate
  table gap (2.5) is confirmed; that it produces a preflight FAIL is not verified here.
* §1.3 prefork ignores the inherited CPU affinity mask (`sysconf` instead of
  `sched_getaffinity`). Source-plausible; not run. If true it also affects every
  containerised deployment on any ISA, so it may deserve its own item.
* §1.4 the Arm rows of `tools/dispatch_expect.json` predate the decoder generation.
* §1.6 `configs/perf/arm-product.json` names no topology and is a generation behind.
* §1.8 `decoder.pre_up_bf16` asks an AVX-512 predicate on Arm; the `region.*` rows resolve
  to `see reason` instead of `OFF`; the KleidiAI region leaf exists and only the region
  BODY is x86-guarded, so that item is body wiring rather than a new Arm kernel.
* §1.9 `QWEN_SD_INT8` opt-in on dotprod and `SD_INT8_BLK=64` were never requalified on Arm.
* §1.10 `tools/flag_parity.py` classifies by the `getenv` site, so a flag read everywhere
  whose kernel is x86-only reads as "all".
* §1.11 the lane silently widens the pool past `--prefork-threads` instead of clamping.

## 4. What NOT to carry over

* **No Arm performance conclusion.** In particular "the decoder lane does not pay on Arm"
  is not established: n = 12, unpaired, cross-process, heterogeneous CPUs, and concurrency 2
  against a 2-slot server — which is below the regime the lane exists for. The engine's own
  note records 14 % swings between processes on an unchanged configuration, the same size as
  every effect reported. The lane must be tried paired, on a homogeneous host, at the
  concurrency it was designed for.
* **`QWEN_SD_CONVT_STACK` neutral on Arm** is from the same probe and is equally
  provisional. It stays default-off on Arm regardless, so nothing depends on it.

## 5. Ordered work, once the VNNI/Turin track is closed

Priority MEDIUM as a track. Item 0 is the exception and is not medium.

0. **Now, not in this track:** the link fix of section 1 plus `SIMD=portable` /
   `SIMD=scalar` link-only CI jobs.
1. Verify §1.2 on an Arm host; if it reproduces, add the KleidiAI gate rows so the Arm lane
   can pass its own preflight. Until then no Arm number is admissible under `ENGINEERING.md`.
2. §1.3 prefork affinity from the inherited mask. Check whether this is Arm-specific or a
   general container defect first; if general, it leaves this track.
3. Documentation and configuration: register the five flags (2.2), refresh the Arm rows of
   `dispatch_expect.json`, give `arm-product.json` a topology, and mark
   `docs/reference-arm-16c.md` with the profile generation it belongs to.
4. §1.11 clamp the lane pool width to the requested `--prefork-threads`. This must land
   BEFORE any lane A/B, or the pool width moves as an uncontrolled third variable.
5. Then, and only then, the lane experiment: paired, interleaved, homogeneous host, at a
   concurrency where there is something to overlap, as a 2x2 against `QWEN_DECODER_BATCH`
   because on x86 the two were adopted together and a single-flip arm cannot attribute the
   result.
6. §1.9 requalify the two Arm decoder defaults.
7. §1.8 region-body wiring, via the existing `.work/decoder-xisa-deferred-track-20260909.md`
   item 2. Not before the rest.

## 6. Relation to the current tracks

Nothing here blocks the Turin qualification of specs 10/11A/12, and nothing here changes an
x86 default. Item 0 is the only cross-cutting one, and it is a build break rather than a
parity question.
