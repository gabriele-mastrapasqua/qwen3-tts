# Current Plan

`ENGINEERING.md` is normative. This is the short current task queue; detailed evidence and
hypotheses live in the linked `.work/*.md` addenda.

## Mission

Ship a CPU TTS server that keeps real-time streams continuous, protects established streams,
and selects only kernels and concurrency points proven by dispatch, parity and full-envelope
streaming evidence. See `docs/serving/boxes.md` for current host qualifications.

## Current objective: legacy CPU v2 streaming

Work is on `feature/old-cpus-simd`, based on main `e391ec5467b0218eeb175f4888ad65b259d1e7c7`.
AVX2 B>1 INT8/Q4 matmat already exists. AVX2 and AVX-512BW no-VNNI B=1 GEMV candidates plus
decoder DL-4 are opt-in. AVX-512F without VNNI reuses AVX2-width kernels for B>1 and checks
F/BW/VL at dispatch and startup. Dotprod Arm uses SDOT; KleidiAI's current packed paths require
i8mm. New candidates stay off pending native-host streaming and performance qualification. Detail:
`.work/legacy-cpu-v2-audit-20260916.md`.

### Completed and validated outcomes

- [x] LEGACY-4 Dispatch/fallback reporting, no-VNNI runtime guard, CPU-safe Linux auto-target and parity contract. Detail: audit §25.
- [x] LEGACY-6 Model-free x86 AVX2 candidate screen and separate physical/SMT modes.
- [x] OLDCPU-DEC-1 AVX2 direct decoder DL-4 signed-widening leaf; compile and Rosetta parity pass, including odd channels/output tails. Detail: audit §24.
- [x] OLDARM-Q4-1 Fused Q4 SDOT matmat candidate for B=2..16; forced M1 parity pass. Default remains off. Detail: audit §24.
- [x] AR-1 Research architecture review — `.work/ar1-post-p2-architecture-review-20260907.md`.
- [x] P2.1 Runtime parity — `.work/p2-cross-backend-runtime.md`.
- [x] MT-1 Playback metric semantics — `.work/professional-streaming-architecture.md`.
- [x] P3.3a Pool capability parity — `.work/p3-runtime-knob-parity.md`.

### Legacy CPU qualification still open

- [ ] ISA-AUDIT-1 Audit released-main ISA/kernel/v2 reachability before new SIMD work. Detail: `.work/isa-v2-reachability-audit-20260919.md`.
- [ ] LEGACY-5 Add/run plain-NEON and dotprod screens on Neoverse N1 and i8mm-capable V1; qualify current v2 streams. Detail: audit §23–24.
- [ ] OLDCPU-1 Measure AVX2 INT8/Q4 B=1 GEMV candidates and v2 stream behavior on a native Zen3/4 or equivalent host.
- [ ] OLDCPU-2 Run complete dispatch/server qualification on AVX-512F without VNNI; compare its AVX2-width path and frequency behavior.
- [x] OLDCPU-4 Add opt-in AVX-512BW no-VNNI INT8/Q4 B=1 GEMV candidates with signed widening, correction and census; cross-ISA compile-checked, adversarial self-tests added, default off. Detail: audit §26.
- [ ] OLDCPU-5 Measure those candidates against AVX2 on an AVX-512F/BW no-VNNI host, including frequency, complete-call and streaming effects. Detail: audit §26.
- [ ] OLDCPU-6 Screen a dedicated AVX-512BW no-VNNI B>1 INT8/Q4 matmat against the existing AVX2 fallback; implement only if a physical host shows a server-level gain. Detail: audit §18, §26.
- [ ] OLDCPU-3 Qualify the direct AVX2 decoder path in complete-call and streaming tests; both `QWEN_SD_INT8=1` and `QWEN_SD_RES1_V2=1` are required.
- [ ] OLDARM-1 A/B INT8 SDOT matmat and fused Q4 SDOT matmat on target CP/Talker shapes; record kernel B separately from server concurrency C.
- [ ] OLDARM-2 Keep KleidiAI dotprod/i8mm splitting blocked until build, runner and RHS packing contracts are independently safe.
- [ ] LEGACY-CPU-1 Run measured AVX2/no-VNNI and Arm dotprod/NEON server screens before changing backend policy.

## Other CPU qualification

- [ ] ARM Qualify BF16 pre-up and CNEXT-I8 as separate paired audio/serving A/Bs; reconcile the Arm profile's `QWEN_SD_BF16_PREUP=1` with its feature-flags description. Detail: `.work/arm-sustained-soak-regression-20260913.md`.
- [ ] AMX Revisit Talker/CP only when real B>=4 work exists; keep CP stateless re-prefill dropped unless new local evidence changes the cost model.

## Playback and server work

Detail: `.work/arm-sustained-soak-regression-20260913.md`.

- [ ] ARM-SOAK-3 Bound admission wall time and reject overload predictably.
- [ ] ARM-SOAK-8 Qualify decoder-lane lifecycle and QoS on the scalable 4x8 topology.
- [ ] ARM-SOAK-5 Measure a concurrent admission/prefill cap.
- [ ] ARM-SOAK-6 Measure active cohort size, phase skew and established-stream preservation.
- [ ] ARM-SOAK-7 Audit ragged temporaries and request allocation churn.
- [ ] OTEL-8 Report which worker answered `/v1/health`.
- [ ] LS-2 Evaluate lead-feedback steady-state quantum after the first chunk.
- [ ] PF-1 Evaluate residual fixed-prompt chunked prefill only with a retained ICL/reference.
- [ ] TQ-2 Verify the fail-fast boundary at a full host.
- [ ] TQ-8 Measure and address leading silence before first speech.
- [ ] EO-1 Revisit shared Talker/CP ownership after P3/P4 coupling is controlled.

## Turin decoder and backend comparison

Detail: `.work/c12-win-track-20260909.md`.

- [ ] C12-WIN-3 Measure short-class fixed cost after WIN-10.
- [ ] C12-WIN-6 Test opportunistic B2 lane batching after higher-priority gates.
- [ ] C12-WIN-6b Freeze AWS campaign order, model matrix and specs 10/11A/12 gates.
- [ ] C12-WIN-7 Run repeated short A/B gates for one mechanism at a time.
- [ ] C12-WIN-8 Qualify the winner with class waves, long/short, Poisson and overload.
- [ ] C12-WIN-9 Measure the C10–C16 capacity curve after the win.
- [ ] QL-2e Calibrate the Turin ceiling at C8/C10 and execute the pre-registered Phase 4.
- [ ] QL-2 Compare promising backends under the same playback-aware harness after a trusted reference.
- [ ] GRAVFULL-2 Complete Graviton5 per-concurrency SOAK qualification.
- [ ] GRAVBOX-1 Ground the GCP c4a highcpu-32 region and pricing candidate.
- [ ] ARMOPT-1 Qualify BF16 pre-up and CNEXT-I8 as separate paired candidates.

## Quantization research (deferred)

Detail: `.work/post-p2-streaming-research-agenda.md`.

- [ ] QP-1 Capture per-layer activation ranges.
- [ ] QP-2 Establish the teacher-forced distance gate before candidates.
- [ ] QP-3 Evaluate per-K-block prefill activation quantization.
- [ ] QP-4 Evaluate BF16 prefix then INT8 prefill.
- [ ] QP-5 Exclude selected worst layers from INT8 prefill.
- [ ] QP-6 Sweep SmoothQuant alpha folded into normalization/projections.
- [ ] QP-7 Measure the KV-seam control arm.
- [ ] QP-8 Revisit `iq4_nl` only if the INT4 track restarts.
- [ ] QP-9 Revisit an offline quantizer only if the INT4 track restarts.

## Deferred decoder and accelerator work

- [ ] DXISA-1 Commonize the winning decoder dataflow only after the Turin C12-WIN checkpoint.
- [ ] DL-1 Test a 4+4 intra-CCX decoder lane only with measured evidence.
- [ ] TQ-3 Review the paired RES1_V2 bank after a valid audio comparison is available.
- [ ] CUDA-11 Close measurement traps in the GPU qualification harness.
- [ ] CUDA-12 Close the retired CP-loop-fusion investigation.
- [ ] CUDA-13 Improve native `k_matmat_bf16` only with a measured roof comparison.
- [ ] CUDA-14 Run the A6000 concurrency ladder.
- [ ] CUDA-16 Keep CPU-only `QWEN_PREFILL_SLICE` disabled on CUDA.
- [ ] CUDA-17 Implement a native GPU prefill as a separate decode path.
- [ ] CUDA-3 Requalify CUDA convolution decoder before enabling it.
- [ ] CUDA-4 Qualify fused Talker+CP batching within B<=8.
- [ ] CUDA-5 Repeat the REPRO-1 A/B/B probe on CUDA.
- [ ] CUDA-6 Design backend-agnostic serving only after backend gates close.
- [ ] CUDA-7 Fix and parity-test the Metal batched path.
- [ ] TQ-7 Verify the CUDA prefork guard on GPU hardware.
- [ ] SL-2 Keep live incremental text/park-not-pad in research scope.

## Qualification gates

A kernel promotion requires parity, observed in-process dispatch/census, and complete streaming
metrics on the target ISA. Report errors/rejects/timeouts, TTFA, STREAM_RTF, required prebuffer,
safe-play-start and stall rates. `STREAM_RTF < 1` alone does not prove continuous playback.
