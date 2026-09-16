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
| 2.6 | `QWEN_SD_RES1_V2` selects on shape, so it also takes res2 | read the branch at `qwen_tts_speech_decoder.c:827` | CONFIRMED, see 2b |
| 2.7 | residual fusion requires AMX; VNNI pays a separate pass | read `cs_conv1d_fused_residual` | CONFIRMED, see 2b |
| 2.8 | AVX2 and AVX-512F have no int8 decoder conv at all | read `qwen_sd_int8_available()` | CONFIRMED, see 2b |
| 2.9 | an undeclared `in_ch <= 768` gate drops every backend to f32 above it | read `qwen_sd_int8_usable` | CONFIRMED, see 2b |

2.3 plus 2.4 is the substantive finding of the review: **the decoder lane, the mechanism of
record on the Turin product profile, ports to Arm Linux unchanged and no Arm profile sets
it** — and the reason it was never tried is most likely a wrong sentence in our own handoff.

## 2b. The residual unit (res1/res2) — verified backend map

A second addendum traced res1/res2 through the dispatcher. Re-checked here; all four
structural claims hold, and they change the shape of the Arm gap.

`causal_conv1d_blas` (`qwen_tts_speech_decoder.c` ~810) picks, in order: AMX bf16 → AMX
Design-D int8 → V2 (DL-4) → v1 int8 tile → f32 im2col + SGEMM.

| build | res1 (k=7 dilated) | res2 (k=1) | residual add |
|---|---|---|---|
| AMX + Design-D | Design-D tiles | Design-D tiles | fused in the epilogue |
| AVX-512 VNNI | V2 (DL-4) | **V2 as well** | separate pass by default; fused only with `QWEN_SD_GLUE=1`, default off and unqualified |
| Arm i8mm/dotprod | v1 dotprod tile, opt-in | v1 dotprod tile | separate pass, always |
| AVX2 / AVX-512F (no VNNI) | **f32 im2col + SGEMM** | **f32** | separate pass |

**(a) `QWEN_SD_RES1_V2` is not "res1 only".** VERIFIED at
`qwen_tts_speech_decoder.c:827`: the branch tests a SHAPE, `kernel >= 1 && in_ch == out_ch
&& (in_ch & 3) == 0`, not a role. It therefore takes res2 (k=1 is square) and every other
square conv in the stack, and `sd_wq_build_v2` is keyed on the weight pointer, not the role.
Both the flag name and the `decoder.res1_v2` dispatch row are misleading, and anyone
implementing the Arm leaf from either would build half of it. This is the single most
useful thing in the addendum.

**(b) Residual fusion is AMX-only, not "x86".** VERIFIED: `cs_conv1d_fused_residual`
returns 0 unless `sd_amx_d_enabled()`. So VNNI pays the separate `signal[i] += c2_out[i]`
pass too — "x86 has it, Arm does not" would be wrong for the residual unit.
**Correction to the addendum:** VNNI is not simply without an answer. `QWEN_SD_GLUE`
(spec 12, `ddfa5d8`) IS the VNNI-side fusion, via `qwen_conv1d_int8_v2_ctx` with a context
and a residual epilogue. It is default off and has never executed on x86, so today the
statement holds; once it is qualified the row changes.

**(c) Three CPU families have no int8 decoder conv, not one.** VERIFIED:
`qwen_sd_int8_available()` (`qwen_tts_kernels.c:8568`) returns 1 only under
`__ARM_FEATURE_DOTPROD` or `__AVX512VNNI__`. On AVX2 and AVX-512F-without-VNNI the whole
residual unit runs in f32. A scalar `qwen_conv1d_int8` is compiled there but unreachable
from serving — that follows from the same predicate, since `use_i8` is false, so the branch
is never taken whatever the leaf contains.

**(d) An undeclared shape gate.** VERIFIED: `qwen_sd_int8_usable` is
`in_ch == out_ch && in_ch > 0 && in_ch <= 768`. Above 768 channels everything falls to f32
on EVERY backend, AMX and VNNI included. No dispatch-map row states this, so a map read as
"int8 ACTIVE" does not mean the wide convs are int8.

### What this means for the Arm work

The gap is one kernel, not a family: a dotprod/i8mm leaf with the DL-4 per-(channel, tap)
layout. The packing side is already ISA-neutral (`sd_wq_build_v2` → `qwen_conv1d_int8_v2_pack`
/ `_cp`) and needs only a different channel padding. The same leaf would serve AVX2 and
AVX-512F, so it closes three families at once rather than one.

One piece of guidance from this side, since the glue contract was written here: an Arm leaf
should be written against `qwen_conv1d_int8_v2_ctx` — context `(tail, tail_cols)` plus the
optional residual in the epilogue — rather than against the plain `qwen_conv1d_int8_v2`
signature. That is the fused form, it costs nothing extra to implement, and it avoids
repeating the two-step history x86 went through. Its exactness oracle already exists and is
ISA-neutral: the `conv1d_int8_v2` self-test cases assert the context+residual path is
bit-identical to the contiguous one.

## 2c. Where Arm decoder time actually goes — and one hypothesis the reviewer killed

REPORTED-MEASURED by the reviewer on their Arm box (open 0.6B weights, `--stream`,
`QWEN_SD_INT8=1`, `QWEN_SD_PHASE=1`, warm 10-frame quantum). Not reproducible here — no Arm
Linux host — so it is recorded with its provenance and not as our own number.

| component | ms | share of `conv_up` |
|---|---:|---:|
| **res1** (k=7 dilated) | 52-56 | **48 %** |
| convt | 26-28 | 24 % |
| snake | 12-15 | 11-13 % |
| res2 (k=1) | 9-10 | 9 % |
| resadd | 4.0-4.2 | 4 % |

The conv stack is ~92 % of the decoder unit (`conv=124.3` of `total=135.0`). If that holds,
**the missing V2 leaf points at the single largest item on Arm**, and item 8 of section 5 is
the right place to spend, not a tidy-up elsewhere.

**A hypothesis proposed, measured and withdrawn — do not chase it again.** The AMX
strip/range "compute only the new output columns" path looked portable: the driver is the
generic `sd_conv_job_t` + pool, already compiled on Arm, and the only coupling to AMX is a
`Wpack != NULL` test that could simply be dropped. The reviewer measured the size of the
prize first: the control path discards **0.4 %** of its output columns at a 10-frame quantum
(1.9 % at 2 frames), and the `ext` build plus the `cut` copy are ~3.0 ms of a 55.5 ms
residual-conv total, about 5 %. So the change buys single digits at best and only at small
quanta. **Do it for tidiness if the code becomes cleaner, never for the number.** Recorded
because it is exactly the kind of plausible-sounding port that gets re-proposed.

The same "portable body, ISA-locked leaf" shape appears three times — CP/Talker regions
(x86-guarded body, Arm KleidiAI leaf already written), the Design-D panel driver, and the
strip/range entries. Useful framing for scoping: what is bolted to one ISA is a tile kernel
and a weight pack, not the scheduling or the dataflow.

## 2d. The September AMX gaps are closed — do not reopen them

REPORTED, structurally consistent with what is readable here (`qwen_region_i8_backend`
carries an `__AMX_INT8__` branch reporting "AMX int8 tiles" at B>=4). The three gaps in
`docs/cross-backend-audit-2026-09-05.md` §2 — regions off at B>=4, batched CP heads off for
the same reason, and the bf16 AMX matmat allocating per call — are all closed on this
branch. Anyone reading that older page should not re-open them.

What AMX still lacks is the V2 residual conv, and that is a **choice, not a gap**:
`causal_conv1d_blas` places Design-D ahead of V2, so on an AMX host with `QWEN_SD_AMX_D=1`
the V2 branch is unreachable by construction. Worth a sentence in the docs so nobody "fixes"
it.

## 2e. GPU serving — one real bug, and a scoping fact for our own roadmap

Not an Arm item; recorded here because it arrived with the same review and needs a home.

**VERIFIED here: `--backend cuda --prefork N` has no guard.** `main.c` creates the resident
CUDA Talker/CP state at :1665 (`QWEN_CUDA_FUSED_TALKER`), and `qwen_tts_serve_prefork` is
called at :3082 — the context is built BEFORE the fork, and a CUDA context does not survive
`fork()`. Children inherit handles they cannot use. A search for any mutual exclusion across
`main.c`, `qwen_tts.c`, `qwen_tts_server.c` and `qwen_tts_cuda.c` returns nothing, and the
transport layer contains zero CUDA references (confirmed: `grep -ci cuda qwen_tts_server.c`
= 0), so nothing downstream catches it either. macOS escapes only by accident, through the
non-Linux `qwen_tts_serve_prefork` stub. This is a silent wrong-answer path, which is worse
than a crash. Fix: refuse the combination, or fall back to the single-process batched server
with a warning, in the same shape as the existing non-Linux stub.

**VERIFIED here: the global GPU seam is bf16-only.** `qwen_tts_backend.h` exposes exactly
`matvec_bf16` and `matmat_bf16`. So `--backend cuda` WITHOUT the fused/resident env vars
offloads nothing on an `--int8` or `--quant-mixed` run — which is what the product profiles
use — while the startup line still advertises GPU offload. That line should say when it will
have no effect.

**Scoping fact that touches the C12-WIN roadmap.** Every mechanism of this generation that
we are qualifying — the decoder lane, `QWEN_SD_INT8`, `RES1_V2`, `GLUE`, `CONVT_STACK`,
Design-D — is a CPU decoder mechanism, and a GPU-resident decoder replaces that component
wholesale rather than tuning it. So **specs 11A and 12 have no value on a GPU lane**, and
none of the Arm decoder work would either. What does carry over is the backend-agnostic
layer: admission, prefix cache, stream layout, `--max-queue`, 503 semantics. Also worth
knowing before anyone quotes GPU readiness: the reported CUDA numbers are throughput and
RTF from an earlier serving generation, and the playback-aware contract this branch
qualifies against (STREAM p95, prebuffer, safe-start, stall@250/@500, soak) has never been
run on a GPU build.

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
8. The DL-4 leaf for dotprod/i8mm (sections 2b and 2c) — aimed at ~48 % of the Arm decoder
   unit if the reported cost map holds. Sized as one kernel against an already
   ISA-neutral packing path, written against the `_ctx` contract, and closing Arm, AVX2 and
   AVX-512F together. Rename or re-document `QWEN_SD_RES1_V2` and the `decoder.res1_v2` row
   first, and declare the `in_ch <= 768` gate in the map, or the next reader repeats (a).

## 6. Relation to the current tracks

Nothing here blocks the Turin qualification of specs 10/11A/12, and nothing here changes an
x86 default. Item 0 is the only cross-cutting one, and it is a build break rather than a
parity question.
