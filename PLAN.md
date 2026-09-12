# Current Plan

`ENGINEERING.md` is normative. This file is the short task queue; reasoning and reviewed
evidence live in the linked `.work/*.md` addenda.

## Mission

Build a CPU TTS server that starts quickly, continuously feeds a real 1x player without
repeated starvation, keeps realtime headroom, protects established streams from new
arrivals and from unrelated slow clients, and only then maximizes sustainable concurrency
and cost per stream. The qualification process discovers the highest concurrency that
satisfies the complete streaming envelope; C4 is not a required operating point.
Rationale and evidence: `.work/professional-streaming-architecture.md`.

## Current trusted state

- Host: GCP c4-standard-24 (12 physical cores, SMT off), 1.7B INT8, decoder Design D
  INT8 AMX with persistent packs, 2x6 prefork, engine-owned pool, batch cap 2.
- Current conservative short q8/threshold2 control: C2/C3 are GOOD; C4 is MARGINAL
  (`STREAM_RTF` p50/p95 0.793/0.856, required-prebuffer p95 596 ms, stall@500 25%).
  `STREAM_RTF < 1` is capacity, not a continuous playback proof.
- Fused-residual Design-D candidate: pooled five-minute C4 SOAK passed the hard stream
  gate in all four windows (`STREAM_RTF` p50/p95 0.8304/0.8933, TTFA p95 526 ms,
  safe-play-start p95 917 ms, zero errors/rejects/timeouts); preferred `<=0.90` was
  missed in one window and per-class p95 was under-sampled. The flag remains default-off.
- Post-C4 capacity screen: C5 is the first NOT STREAMABLE point under the complete
  envelope (STREAM p95 0.852 but TTFA p95 4.44 s and safe-play-start p95 4.60 s); C6
  shows the same failure. F2 causally decomposed the C5 tail: under cap 2, three full-wave
  requests waited 3.6–4.9 s before parent `accept()` and >97% of client-to-first-PCM
  elapsed before engine admission; `--max-queue 0` converted the same overload to 3
  immediate parent-side 503s. A secondary child/engine queue + prefill term remains, but
  is sub-second. Do not advertise C5/C6 as realtime capacity. Detail:
  `.work/p4-prefork-admission-bound-20260907.md`, `.work/f2-c5-startup-decomposition-20260908.md`.
- CT-1 confirms prebuffer follows quantum (q8 ~0.7 s p95 in short SOAK; q32 ~2.5 s)
  while RTF changes less. q32 is rejected as a production streaming policy.
- Decoder MACs already run on real AMX with wide N; its wall is glue (im2col, quantization,
  ~41 rendezvous and ~110 BLAS calls per call, snake, tails). AMX can touch at most
  ~10-20 % of request wall; more tile tasks regressed (M split rejected).
- Per-worker effective batch ~1.1-1.3 at C4: Talker/CP run as DRAM-bound B=1 GEMV, weights
  read per worker. The 1x12/batch-4 probe measured decode-burst coupling, not Talker
  batching, and is not evidence against a single engine.
- Inline prefill stalls every established stream 108-240 ms per admission; a slow client
  blocks its worker's engine thread (blocking writes, no send timeout).
- LS-4 utilization-aware admission was falsified on the same host: thresholds 40/60/80 ms
  admitted all tested fifth arrivals, but established STREAM_RTF p95 stayed 0.985-1.004,
  stall@250 was 50%, and post-admission max-gap p95 reached 653-704 ms. Keep the
  diagnostic default-off; cap2/q4 remains the reference. Detail:
  `.work/ls4-utilization-aware-admission-20260908.md`.
- Harness (2026-09-07): one metric core `tests/playback_sim.py` with per-request
  safe_play_start, fixed-buffer stall rates, max_gap, coalesced-read share; marks are
  client-observed. Batched synchronous streams now publish the header at admission and
  accepted sockets set `TCP_NODELAY`; PCM writes remain synchronous unless OUT is enabled.
  Detail: `.work/mt4-transport-boundary-20260907.md`.

### Current 8-core AMX product decision

- The current 8-physical-core reference is `1x8@0-7`, SMT off, Design-D INT8,
  fused residual, warm strip, q4, engine pool and fail-fast admission. The
  strict `amx-product` profile now explicitly enables the official known-text
  SL-1 layout (`QWEN_TTS_STREAM_LAYOUT=1`); ICL/clone and live incremental text
  are outside this lane.
- 1.7B: **C2/cap2 is the highest full-envelope GOOD point**. C3 is a healthy
  short/isolated-bank screen but not a full production point because the
  corrected C3 SOAK still has tail drift and pooled STREAM_RTF p95 just over
  one. Detail: `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- 0.6B: **C3/cap3 is the highest full-envelope GOOD point** on this host. C4 is
  a non-promoted screen; C5 is the first clearly bad short-bank point. Detail:
  `.work/gcp-c4-highcpu16-amx-product-capacity-20260908.md`.
- These are playback-aware product points, not a claim that the 8-core host can
  sustain C4 or that low-rate Poisson probes establish an economic rate. Cost
  per good stream remains UNKNOWN without grounded pricing.

## Immediate priorities

### P0 Metric truth — detail: `.work/professional-streaming-architecture.md` E1, E8, E11, E12

- [x] MT-1 Receive-mark semantics audited; TTFB stamped independently of TTFA
      (`header_to_audio_ms`); coalesced-read share reported per run — detail:
      `.work/professional-streaming-architecture.md` E12.
- [x] MT-2 Per-request `safe_play_start`, stall_rate/stall_ms @100/250/500/1000 ms,
      max_gap; summaries in the wave and soak analyzers; `tests/test_playback_sim.py`.
- [x] MT-3 Superseded readings corrected in `docs/serving-operations.md` section 5,
      `docs/BENCHMARKING.md` sections 7-8, `ENGINEERING.md` section 9, AWS reference notes.
- [x] MT-4 Runtime transport boundary: batched streams send the header before synthesis and
      accepted sockets use `TCP_NODELAY`; server/client event ordering is proven on the
      continuous path. Per-chunk flush tracing remains optional and client marks remain
      client-observed. Detail: `.work/mt4-transport-boundary-20260907.md`.

### P0 C12 preferred gate on the frozen Turin architecture (ladder paused after bounded falsifiers) — detail: `.work/c12-win-track-20260909.md`, checkpoint: `.work/c12-win-checkpoint-20260909.md`

Goal: sustain C12 with the full streaming contract AND STREAM_RTF p95 <= 0.90 including
the short/conversational soak tails (today waves 0.82-0.85, soak pooled 0.912, short
0.959, conversational 0.914; cadence already good). One mechanism at a time against the
frozen `turin-c8a-32c-vnni-product` control; lever of record: decoder residency down ->
CP-overlap share down -> sustained tail down. Codex owns implementation; no push unless asked.

      Review 2026-09-09 (`.work/c12-architecture-review-20260909.md`, read-only): the
      remaining unit is half f32 BLAS (transformer/convnext/init ~12 ms + convt 8.6 ms,
      ~330 MB f32 weights per unit, excluded from int8 by construction) plus ~8 ms of
      copy/calloc glue; closed-loop admission (inline prefill) is the likely short-class
      tail; width is bounded out; phase-aware placement cannot help at sustained B3.
- [x] C12-WIN-1 Zero-code discriminators (2026-09-09): helper is NO-GO as an implementation
      (TTFA p95 172 -> 683 ms, safe-start 417 -> 922 ms, stall@250 appears) but it CONFIRMS
      the mechanism: with prefill off the loop the short class drops 0.966 -> 0.915 and
      pooled 0.923 -> 0.915 — inline admission is the short tail, worth ~0.05; fixed B3
      shows 54.9% decoder-overlap wall and CP 22.2 -> 36.5 ms median; spin 4096/16384 does
      not beat the 65536 control. Decoder unit 49.2 ms by cost map (conv_stack 43.3,
      transformer 5.2). Detail: `.work/c12-win-step1-3-20260909.md`.
- [x] C12-WIN-1a Pre-upsample BF16 diagnostic (2026-09-09): persistent BF16 weights and
      matmat path implemented/default-off. The bounded Turin server screen moved modestly
      (STREAM p95 .842 -> .825), but the same-generation paired audio gate failed
      (`mel_corr=.97890 < .98`) and the non-clean B3 diagnostic did not show lower decoder
      residency or CP overlap cost. Keep BF16 default-off; detail:
      `.work/c12-win-bf16-preup-20260909.md`.
- [x] C12-WIN-2 Decoder-residency falsifiers, first round (2026-09-09): BF16 pre-up,
      ConvT one-GEMM, allocation-only glue, and VNNI RES1_V2 split-input were each
      isolated; no candidate earned a serving A/B. These are IMPLEMENTATION verdicts:
      the tested BF16 arm covered the transformer only (5 ms of the unit) with bf16
      activations; the ConvT arm used a zero-expanded input panel (k× the FLOPs), not the
      proposed one-GEMM-per-layer on the un-expanded input; the glue arms were two pieces
      run separately. Untested: weight-only bf16/int8 for the conv_stack f32 weights
      (convnext pw, initial conv, convt ≈ 190 MB/unit) and the combined glue removal.
      Forensic audit: `.work/c12-win-forensic-audit-20260909.md`.
- [x] C12-WIN-2a ConvT one-GEMM falsifier (2026-09-09): exact decoder batch parity passed,
      but the expanded f32 panel made the treatment 14–32% slower across B1–B4/chunk 1–8.
      Rejected and reverted; no server A/B. Detail: `.work/c12-win-convt-one-gemm-20260909.md`.
- [x] C12-WIN-2b VNNI glue/preparation falsifiers (2026-09-09): allocation-only and
      split-input V2 were exact/parity-safe where tested but slower or neutral; both
      were reverted. The remaining alternative geometries are not justified by the
      current evidence. Detail: `.work/c12-win-glue-vnni-20260909.md`.
- [x] C12-WIN-10 Admission slicing (prefill as resumable token-range slices inside the frame
      loop). Spec: `.work/c12-win-admission-slicing-implementation.md`; local evidence
      `.work/c12-win-admission-slicing-20260910.md`. CLOSED 2026-09-10 on Turin: state parity
      CORRECT, serving behaviour a severe REGRESSION, flag stays default-off and unpromoted.
      Sliced-state parity was proven exact for every partition without a one-token slice, and
      two engine defects were found and fixed while proving it. The server A/B says the
      mechanism must not be promoted. Closed-loop C12, 10 minutes per arm, frozen profile,
      one variable: completed 1294 -> 721 (-44 %), TTFA p95 183 -> 1574 ms, STREAM_RTF p95
      0.922 -> 4.609, and 7 server request timeouts against 0 on the control. It made the
      established-stream interference it was built to remove about five times worse.
      Note what it did NOT test: all four workers report mean_slices=1.00, so with a warm
      prefix cache the admission prefill is ~1 new token (seq_len=10, prefix=9) and nothing
      was ever actually split. The damage therefore comes from the sliced-admission PATH, not
      from slicing -- most likely the one-admission-per-frame-iteration break stealing
      iterations from running streams. A first A/B attempt with the true-wave arrival model
      was void and is not cited: with a positive flag and n_active==0 the code takes the whole
      prefill in one slice by design, so a wave that releases every request into an idle
      engine cannot reach the mechanism at all. Any retry needs a redesign of the admission
      path first, plus a cold-prefix workload so real multi-slice prefills occur.
- [x] C12-WIN-11 A Conv-stack traffic: ConvT as ONE un-expanded GEMM per layer with a fused
      two-tap carry/bias epilogue. Spec: `.work/c12-win-conv-stack-implementation.md`.
      CLOSED 2026-09-10 on Turin: implementation CORRECT, effect NULL, flag stays default-off.
      Correctness passes on x86 -- self-test 0 failures with the convt_stack cases at 3e-8..1e-7
      against a 1e-5 contract, dispatch verified ON/OFF, CLI audio mel-corr 0.99962 at identical
      duration. The microbench (1.7B, 4 threads, taskset 4-7, B1-B4, 9 warm reps) shows no
      effect at the product quantum: chunk-4 deltas B1 -0.20, B2 -0.70, B3 +0.70, B4 +1.30 ms,
      and 16 of 32 cells faster -- a coin flip. A control-vs-control run of the SAME arm on the
      same binary in the same minutes measured a noise floor of -0.40..+2.70 ms at chunk 4 and
      up to 32 ms at chunk 16, so every one of those deltas is inside the noise. Note for any
      future rung: the >= 2 ms gate this spec asked for is BELOW this harness's own resolution
      at B3/B4 (noise alone is +2.7 ms there); a rung that needs to resolve 2 ms needs paired
      replicates, not a single run of each arm.
- [x] C12-WIN-12 VNNI glue as one combined change: out-of-place snake1, V2 kernel with
      (tail, tail_cols) context and residual epilogue, plain allocs, ownership transfer.
      Spec: `.work/c12-win-vnni-glue-implementation.md`. CLOSED 2026-09-10 on Turin:
      correctness REPAIRED, performance NO-GO, flag stays default-off and unpromoted.
      The first x86 `--self-test` failed 10 cases, all the `ctx+residual` contract: with
      -ffast-math the compiler re-associates the epilogue's four-term sum only when a
      residual is supplied. Not benign here, because the next residual unit re-quantises
      per position, so one ulp shifts amax and the whole position's scale: 115 LSB on a
      9550 peak end to end. Fixed in `d49aa10` by disabling reassociation for
      `sd_dconv_worker` alone -- self-test 10 failures to none, `QWEN_SD_GLUE=0/1`
      byte-identical, and the attribute costs nothing on the default path (control vs
      control -2.4..+2.9 ms, no systematic sign). With the epilogue exact the MECHANISM
      is slower than its control in 21 of 32 cells: +0.8 ms at B3 chunk 4 against a gate
      asking for -3 ms, rising to +60 ms at B4 chunk 16. The -11 ms seen before the fix
      was measured while the fused path was still free to re-associate, so it was not
      computing the same result as the control and was never a valid comparison.
- [ ] C12-WIN-3 Short-class fixed cost: only after WIN-10: ramp 1,2,4 (control) vs 1,4
      (vs 2,4 only inside the TTFA gate); short + conversational playback metrics. No q8.
- [x] C12-WIN-4 Old preparation flags: DIRECT_DWCONV/INPUT, STRIP, FUSED_RESIDUAL are
      inert on VNNI (AMX-D gated); the VNNI glue falsifiers are now closed. DIRECT_CONVT
      is superseded by the rejected one-GEMM falsifier.
- [x] C12-WIN-5 Phase-aware decoder overlap: NO-GO at the measured pinned B3 overlap
      share of 54.9%; no asymmetric Talker/CP width mechanism is justified.
- [ ] C12-WIN-6 Opportunistic B2 lane batching (optional, last): residency of 2 units vs 2
      requests, decoder off the critical path, mailbox bounded, reject on any cadence loss.
- [ ] C12-WIN-6b AWS campaign order, model matrix and gates for specs 10/11A/12:
      `.work/aws-qualification-checklist-20260910.md`. PRIMARY qualification path is
      1.7B Base OSS + Galatea qvoice (clone conditioning), SECONDARY control is 1.7B
      CustomVoice + Ryan (preset-speaker). The public ~25 MB CC0 grafts load on Base with
      `--load-voice ... --icl-only`; what is NOT yet exercised is the clone conditioning
      path through the BATCHED SERVER, which Phase A0/A confirms before any timing claim.
- [ ] C12-WIN-7 Short A/B gate per candidate (control vs one mechanism, repeated short C12
      waves, playback-aware metrics, gain > noise) before any soak.
- [ ] C12-WIN-8 Qualify the winner: C12 class waves, long+short, Poisson, overload
      unchanged, 30-min soak by class and 5-min window; STREAM p95 <= 0.90 overall and per
      class, stall@500 0, cadence targets kept; report an exact boundary rather than move the gate.
- [ ] C12-WIN-9 Capacity curve C10-C16 after the win, classified as preferred /
      mandatory-qualified / hard-capacity (never one "max C" number).
- Stop: if no target, no falsifier and no screen moves C12 above noise, hand the evidence to
      the post-Turin architecture review instead of stacking micro-optimizations.

### ARM-LINUX-V2 — parity implementation complete; optional policy qualification deferred

- [x] Arm Linux serving is at the v2 generation on `feature/arm-parity-vnni` (`6117437`).
      The implementation, exact self-tests, ISA/link checks, dispatch checks and final
      config policy are complete. The track document with the verified/unverified split
      and do-not-carry-over list remains `.work/arm-linux-v2-parity-track-20260910.md`.
      Headline finding, CONFIRMED against this tree: the decoder lane
      (`QWEN_SD_LANE_SPLIT` / `QWEN_SD_LANE_ELASTIC`) has NO ISA guard — only `__linux__` —
      so the mechanism of record on the Turin product profile ports to Arm unchanged, and no
      Arm profile sets it. The reason it was never tried is a wrong sentence in our own
      handoff, corrected 2026-09-10. Also confirmed: the five newest decoder flags have zero
      entries in `docs/feature-flags.md`, and `g_mm_gate[]` has no KleidiAI int8/bf16 rows.
      The old unpaired n=12 probe on a heterogeneous box at concurrency 2 against a 2-slot
      server remains non-evidence. The new exact-commit Axion screen is recorded below as
      a one-wave performance screen only, not as an Arm product qualification.
      Ordering: the build break above is NOT part of this track and must not wait for it.
      Progress on `feature/arm-parity-vnni` (2026-09-10): items 0 (link fix, = TQ-6), 1
      (KleidiAI gate rows), 2 (prefork plans on the inherited mask), 3 (docs + expectation
      rows), 4 (lane honours the requested engine width) and the Arm DL-4 leaf of item 8
      are implemented; the leaf passes the 20-case `--self-test` on aarch64 dotprod.
      Item 7's region body is wired on Arm through the prepared-state API that was written
      for it and never connected: Talker and CP batched regions now pack the KleidiAI LHS
      once per projection group and run the same kai_i8_task in-region.  Verified on the
      16-core Neoverse-V2: Talker region 12/12 WAV byte-identical on/off, CP region 12/12
      byte-identical, arm-product preflight VALID, dispatch gate PASS; C10 2x8 lane4
      elastic + RES1_V2 + GLUE + CONVT_STACK measures STREAM p95 0.843 against 0.939 for
      the untreated tree (WAVE screen, no SOAK yet).  RES1_V2 audio gate: 21/21 paired
      files, mel-corr min 0.9945.
      The original next list is now closed at implementation level: pre-transformer BF16
      wiring, rectangular/wide DL-4, and multi-slot DL-4 are all implemented and tested.
      The lane-team constraint is handled by the prepared-state prep/run pair (tid/nt),
      while the region/prepared-state API remains keyed on the ORIGINAL f32 weight pointer.
      DONE since: DL-4 rectangular/wide shapes (API `in_ch`/`out_ch`, any shape when the flag is
      on; two rectangular self-test cases exact / 5.6e-3); ConvNeXt pointwise pair on KAI
      int8 (`QWEN_SD_CNEXT_I8`, default off) --
      6 paired server texts mel-corr min 0.99736 / mean 0.99805, C10 0.843 -> 0.821.
      Item 1 implementation is now wired through full, streaming and ragged pre-transformer
      forwards: Arm KAI registers all persistent BF16 rows and unregisters them on teardown;
      the Neoverse-V2 smoke is functional on both 0.6B and 1.7B. The corrected Graviton5
      prepared-LHS micro A/B and paired C1 WAV gate are now PASS, while the broader BF16
      product promotion screen remains open (the implementation is default-off). Item 3 is implemented for VNNI and
      Arm SDOT with compact and production strided APIs, exact S=2/S=3 oracles, and a lane
      cohort. The Arm 2/3-slot WAVE reached group=2/3 with zero mailbox overruns, but measured
      2.8--3.9% slower on the short 0.6B/1.7B A/B, so it is also default-off. Evidence and
      remaining qualification gaps: `.work/arm-linux-v2-parity-implementation-20260911.md`.
      Exact-commit Axion FAST screen (Neoverse-V2, 2x8, short synchronized wave, custom
      1b7 model, INT8) reached C8 with lane split=4: C6/C8 STREAM p95 `.646/.716`,
      TOTAL p95 `.699/.806`, TTFA p95 `236/303 ms`, zero errors/rejects; C12/C14 are
      screen-only and miss playback headroom. Inline control was `.917/.860` STREAM p95
      at C6/C8; split=2 was slower, so no lane split is promoted in the Arm profile.
      This is not an apples-to-apples Turin claim: Turin has 32 cores and the reference
      screen uses the open 1.7B model. Turin's 4x8 screen was `.87/.87` STREAM p95 at
      C6/C8, making the Arm C6/C8 steady-state screen comparable despite half the cores;
      first-audio and full qualification still need a repeated product run.
      Arm cost map (REPORTED-MEASURED, not reproducible here): res1 is ~48 % of the upsample
      convs and the conv stack ~92 % of the decoder unit, so the missing V2 leaf aims at the
      largest single item. DO NOT chase the AMX strip/range port: it was measured first and
      discards only 0.4 % of columns at a 10-frame quantum (~5 % of residual-conv time). The
      three September AMX gaps are CLOSED on this branch; do not reopen them from the older
      cross-backend audit page. AMX lacking V2 is a dispatch-order CHOICE (Design-D precedes
      V2), not a gap.
      Follow-up 2026-09-11: `c6e6e26` shares the KleidiAI activation preparation across
      Talker/CP region workers (same prepared-state kernel); `5b03269` makes the BF16 KAI
      pre-up prepare synchronously before dispatch, removing a barrier that was unsafe for
      serial/GCD/narrowed pools. Mac build/self-test and a clean Graviton5 build/self-test
      pass; four Graviton5 C4 WAVs are byte-identical to the pre-change baseline. The
      Graviton5 4x8 all-on FAST screen is exploratory only (C8/C12 TTFA p95 257/328 ms,
      STREAM p95 0.794/0.853, zero errors/rejects); it does not close the Turin regression
      gate or qualify the Arm product profile.
- [x] ARM-LINUX-V2 item 8: the residual unit (res1/res2). VERIFIED backend map in
      `.work/arm-linux-v2-parity-track-20260910.md` section 2b. Four facts the dispatch map
      does not show: `QWEN_SD_RES1_V2` selects on SHAPE (`kernel>=1 && in_ch==out_ch &&
      !(in_ch&3)`), so it takes res2 and every square conv, not just res1 — implementing
      from the flag name builds half of it; residual fusion needs AMX, so VNNI also pays a
      separate pass (`QWEN_SD_GLUE` is the VNNI answer, default off, unqualified); AVX2 and
      AVX-512F-without-VNNI have NO int8 decoder conv at all, so the gap is three CPU
      families; and an undeclared `in_ch <= 768` gate drops every backend to f32 above it,
      AMX and VNNI included. Work: one dotprod/i8mm DL-4 leaf against the already ISA-neutral
      packing path, written to the `qwen_conv1d_int8_v2_ctx` contract. DONE on
      `feature/arm-parity-vnni`: the leaf exists for Arm dot-product and the API is now
      rectangular (`in_ch`/`out_ch`, Cp from `in_ch`), so DL-4 also takes the initial/pre
      convs and the wide channels that the v1 panel and Design-D paths cannot; --self-test
      covers both rectangular shapes and the 20 square ones. AVX2/AVX-512F-without-VNNI stay
      on the f32 fallback, so the three-family claim of this item is not delivered. The flag
      and the `decoder.res1_v2` row are re-documented but not renamed. Arm widened-path
      quality/perf promotion remains intentionally open; parity implementation and exact
      Arm/x86 build/self-test gates are complete (the VNNI kernel shares the API change).

- [x] ARM-LINUX-V2 final config TODO, completed last after the BF16/multi-slot A/B and x86
      VNNI compile/parity checks: update `configs/perf/arm-product.json` and
      `configs/perf/axion-16c-ttfa.json` with RES1_V2, lane, multi-slot and BF16 policy.
      RES1_V2 is available; BF16 pre-up and multi-slot remain explicit default-off controls
      until their separate quality/16-core qualification gates pass.

- [x] VNNI DL-4 multi-slot promotion smoke (2026-09-11): after the exact multi-slot oracle
      fix, the Turin product A/B ran three synchronized short waves at C4/C8/C12/C16 and
      two mixed short/long waves at C8/C12 against the explicit `QWEN_SD_MULTISLOT=0`
      control. All 120 requests per arm completed with zero errors/rejects; sustained
      stream/total p95 and req/s improved coherently at C8-C16. The Turin product profile
      now defaults `QWEN_SD_MULTISLOT=2`; the control profile pins 0, Arm/KleidiAI stays
      default-off pending its own 16-core/quality gate, and paired audio quality remains
      required before calling the feature qualified across products.

- [x] PRE-GRAVITON-5 Turin regression applicability gate: the Turin VNNI product/control
      A/B and the C4/C8/C12/C16 multi-slot smoke are already recorded in the Turin handoff
      and the preceding VNNI promotion work. The later commits `c6e6e26` and `5b03269` touch
      only the KleidiAI/Arm paths: on x86 VNNI the new region helpers are not selected and
      the BF16 KAI consumer is a fallback no-op. Therefore Turin does not need another run
      solely for this Arm-only delta. Reopen this gate if shared x86 kernels, threading,
      profiles, or dispatch code change. The Turin checkout remains a dirty bench checkout:
      sync only tracked source/config/commits, never models/private/WAVs, and keep the
      privacy/log/tree check in force.

- [x] GRAVITON-5 v2 mini-sweep and flow audit (2026-09-11): clean 32-core Neoverse-V3
      build/self-test/doctor passed; 4x8 was the useful topology. The exploratory all-on
      Arm v2 screen held C12/C16 at STREAM p95 `.834/.947` and TTFA p95 `347/470 ms`,
      while C18 crossed the edge (`1.514` STREAM, `529 ms` TTFA). Lane+multi-slot was the
      main gain; BF16-only was marginal and remains default-off. The marker follow-up now
      attributes the decoder panels: C2/C4 cost-map parity is still PASS, with 8/8 workers
      entered and 100% panel occupancy at both levels; the previous UNACCOUNTED row was
      instrumentation, not an inactive kernel. `conv_stack` is 93.0%/91.4% of the serve
      decoder map and pool wait is a real 16.2%/24.1% completion-wait share. The prepared
      BF16 LHS reuse is wired through full/streaming/ragged decoder paths; corrected C12
      FAST A/Bs improve STREAM/TOTAL p95 directionally in both orders with zero errors, and
      paired C1 WAVs are byte-identical. Pool-spin 0/4096/16384/65536 was noisy, so the
      Arm 65536 default remains. Scratch stats showed zero spills and grow-once/reused
      arenas. Full qualification remains open; details:
      `.work/graviton5-arm-v2-mini-sweep-20260911.md`.

- [x] GRAVFULL-1 Graviton5 qualification campaign execution (2026-09-11): the clean `arm-product`
      RES1_V2/KAI INT8 baseline was built and dispatched on the selected 4x8 topology;
      doctor, strict preflight, caps/dispatch/self-test, CPU check, paired structural
      audio, capacity waves and C4 SOAK passed with zero errors/rejects and zero fixed-
      buffer stalls. The all-on arm also passed serving/resource SOAK and the 1.7B/0.6B
      FAST ladders, but its paired mel gate failed (`0.88559` minimum vs `0.98`), so
      BF16 pre-up and multi-slot remain default-off. This closes the campaign execution,
      not the per-concurrency capacity qualification; exact evidence:
      `.work/graviton5-arm-v2-full-qualification-20260911.md`.

- [ ] GRAVFULL-2 Graviton5 per-concurrency SOAK qualification: when the box is leased
      again, split the matrix by model. For 1.7B, run identical closed-loop SOAKs for
      control OFF and exploratory all-on at C6/C8/C12/C16; run C18 as a diagnostic edge
      only if admission remains meaningful (C4 is already covered for 1.7B). For 0.6B,
      extend the existing C1/C4/C8/C12 FAST screen through C16/C20/C22/C24+ until the
      knee, then SOAK the selected levels for both arms. Keep the current fail-fast
      admission/batch cap as the control, but add a second small-model pass with an
      explicitly raised per-worker batch/admission cap when testing C20/C22/C24; record
      the exact cap and queue policy in the manifest. A WAVE is not a capacity
      qualification: do not call any 1.7B C12/C16/C18 or 0.6B C16/C20+ level qualified
      without same-model SOAK evidence and errors/rejects=0. A c4a 32-core Arm result
      can select candidate C levels and cap settings for this pass, but cannot replace
      same-host Graviton5 evidence. All-on remains non-promotable unless paired audio
      also passes. After the current c4a campaign, if the 1.7B C12-C16 interval is
      incomplete or C16 fails at the current cap, add a fine-grained C13/C14/C15
      sweep for control and all-on. Run it first at the existing cap for comparability;
      if admission rejects are the limiting factor, repeat the selected levels with an
      explicitly raised cap and label that as a separate admission experiment.
      The c4a candidate admission experiment is now complete (2026-09-12): with
      batch-cap 8 per worker on 4x8, WAVE admission reached C32 with zero rejects for
      both 0.6B arms and C36 rejected 12; no raised-cap C20+ SOAK passed the strict
      playback/KPI gate. This selects C20 as an exploratory all-on candidate and C32
      as an admission-only candidate, not as qualification. Detail:
      `.work/c4a-arm-v2-raised-cap-report-20260912.md`.

- [ ] GRAVBOX-1 GCP c4a highcpu-32 Arm candidate: record the Iowa region and quoted
      `$1.21/hour` cost, then—only after the per-model SOAKs and feature gates—derive a
      separate 32-core Arm profile from `arm-product` with the newly qualified flags.
      The box setup alone must not promote BF16 pre-up, lane, multi-slot or other
      optional features; keep the profile explicitly tied to its 32-core topology.
      A host/model-scoped all-on deployment candidate is now recorded at
      `configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`: C16 preferred, C20 soft edge,
      C32 admission-only. It remains `unqualified` until a clean-tree canonical SOAK
      and paired 0.6B audio gate are run.

- [ ] TURIN-POST-ARM quick regression safety screen (before the next cross-ISA release
      claim or paid capacity ladder): rerun the current VNNI control and yesterday's
      promoted feature profile at C6/C8, including one short FAST and the existing
      TTFB/TTFA/STREAM/TOTAL comparison. The Arm-only applicability gate says this is
      not required to attribute the Arm result, but the screen is retained as the
      requested regression check; record any drift separately from the Arm report.

- [ ] ARM optional-feature promotion: qualify BF16 pre-up, multi-slot and CNEXT-I8 as
      separate paired A/Bs only after the Graviton baseline. BF16 and multi-slot remain
      default-off: the existing Arm quality/perf evidence is not a promotion, and the
      current multi-slot short A/B was negative. Do not bundle these into the baseline
      claim or change the Arm JSON defaults without paired audio plus serving evidence.

### Deferred DECODER-XISA — converge decoder dataflow after C12-WIN

- [ ] After the Turin C12-WIN track reaches a stable checkpoint, commonize the winning
      streaming-decoder dataflow across x86 VNNI, x86 AMX and Arm/KleidiAI; keep this
      deferred and do not mix it into the current paid Turin ladder. Start with a short
      design/dataflow audit, then parity-gated leaves in this order: RES1_V2 direct causal
      convolution, common glue/materialization removal, pre-upsample BF16/INT8 matmat,
      and one-GEMM ConvT. Detail and gates: `.work/decoder-xisa-deferred-track-20260909.md`.

### Deferred QUANT-PTQ — calibration-aware quantization revisit (MEDIUM/LOW)

- [ ] Revisit lower-precision prefill and weight storage using calibration/optimization-aware
      PTQ (AutoRound-style or equivalent) instead of the engine's earlier straightforward
      conversion. Production keeps prefill in BF16 deliberately; the earlier simple INT8
      prefill and simple INT4/Q4 attempts were rejected because pronunciation and speaker character
      drifted audibly while the audio stayed otherwise valid. Those verdicts reject THOSE
      IMPLEMENTATIONS, not lower precision as a direction -- do not record "INT8 prefill" or
      "INT4" as architecturally disproven. Tracks: calibrated INT8/W8A8 prefill; quality-
      optimized Q4/INT4 or mixed precision for suitable Talker/CP/prefill regions; offline
      calibration only, with the C runtime consuming packed weights and scales and no
      training machinery; and a re-test of whether the V2 kernels change the premise. GATE:
      a performance gain is irrelevant unless pronunciation and speaker character survive against the current INT8 + BF16-prefill baseline, judged by paired audio, ASR
      and listening -- waveform/mel/duration equality is necessary and not sufficient, since
      the earlier rejections passed exactly those. Start only after the C12-WIN items and
      the report qualification work. Detail: `.work/quantization-ptq-revisit.md`.

### P1 Cadence truth (current binary, Tier A only) — detail: `.work/p1-cadence-truth-20260907.md`

- [x] CT-1 Quantum discriminator at C3/C4, including gang-off control; q32 is rejected.
- [x] CT-2 Decoder intercept/slope and `[SDPHASE]` attribution; SQ-1 remains GO.
- [x] CT-3 Inline admission interference measured with matched control; LS-4 remains P3.
- [x] CT-4 Talker B1/B2 measured; EO-2 remains viable (B2/B1 step ratio ~1.10).
- [x] CT-5 C2/C3/C4 playback envelope: GOOD / GOOD / MARGINAL.

### P2 Small-quantum decoder — CLOSED checkpoint: `.work/p2-checkpoint-20260907.md`

- [x] SQ-1 Bounded warm range slice: newly produced columns use direct INT8 A preparation
      and persistent Design-D B packs in the serving reference. The complete
      strip → snake → conv1 → snake → conv2 → residual executor is not implemented and
      moves to AR-1 as an architectural candidate.
- [x] SQ-2 Bounded fixed-cost audit: default-off slices cover direct streaming/ragged
      ConvT, depthwise and warm-input preparation; fused residual remains a candidate.
      Direct one-row gather/quantization and BLAS-C residual were rejected and reverted.
      Details: `.work/p2-checkpoint-20260907.md` and the linked experiment addenda.
- [x] SQ-3 Decoder AMX reachability and scoped accounting recorded; the four whole-request
      quantities are not fabricated where the current evidence has no valid denominator.
      Detail: `.work/p2-checkpoint-20260907.md`.


### AR-2 reviewed order — CLOSED docs checkpoint

- [x] AR-1 implementation audit and AR-1b external/model supplement are frozen against
      the P2 HEAD: `.work/ar1-post-p2-architecture-review-20260907.md`,
      `.work/ar1-codex-implementation-audit-20260907.md`,
      `.work/ar1b-external-research-supplement-20260907.md`.
- [x] AR-2 verified the official known-text dual-track layout against the C prompt/step
      path and froze the implementation order: `.work/ar2-sl1-semantics-20260907.md`.
      Prefix-cache reuse is not resumable prefill; q1/q2/q4 are smaller complete decoder
      calls, not intra-call preemption; whole-request AMX wall remains UNKNOWN.

### P3 Serving cadence and first-play

- [x] OUT-1/OUT-2 Bounded per-stream PCM queue, detached non-blocking writer, byte/memory
      cap, timeout, cancellation/disconnect semantics and slow/stopped-reader tests are
      implemented behind `QWEN_SERVER_ASYNC_OUTPUT=1`; C1/C2 path, matched C3/C4 Tier-A
      integration, byte-identical audio and slow-reader gates pass. It remains default-off:
      the C3/C4 wave shows no material KPI change, and longer-concurrency thread/memory
      qualification is still open. Engine enqueue and transport-write timestamps remain
      distinct. Detail: `.work/stream-output-isolation-20260907.md`.
- [x] SL-1 Known-text official dual-track layout implemented behind
      QWEN_TTS_STREAM_LAYOUT=1 and carried through CLI, batch and continuous-server
      admission paths. The known-text Ryan/English lane passed current-generation
      structural/audio, prefill-scaling and 8-core server interference gates and is
      explicit in `amx-product`; ICL/clone and live incremental text remain outside
      the lane. Detail: `.work/sl1-known-text-stream-layout-20260907.md` and
      `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- [x] LS-1 Minimal credit-gate skeleton implemented and falsified at C3/C4 behind
      `QWEN_STREAM_LEAD_GATE=1`: first audio remains eligible, but hard suppression at a
      250 ms target parks ~95.8% of checks, lowers useful worker work and does not improve
      stall rates. Keep default-off; do not add EDF/LS-2 on this realization without a new
      mechanism. Detail: `.work/playback-lead-gate-fc-20260907.md`.
- [x] LS-3' Small complete decoder calls at safe existing boundaries; q1/q2/q4/q8 floor
      established in a Tier-A C3/C4 screen. q1 is rejected; q2/q4/q8 remain policy
      candidates and no intra-call preemption is claimed. Detail:
      `.work/decoder-quantum-floor-20260907.md`.
- [ ] LS-2 Lead-feedback steady-state quantum: first chunk remains one frame, bounded lead
      window, explicit minimum efficient quantum; q8 remains the upper control until proven.
- [ ] PF-1 Residual fixed-prompt chunked prefill only if a retained ICL/reference or
      non-streaming mode still leaves a genuinely long prefix after SL-1. It is not a
      blocker for the current known-text 1.7B product point; the cloned-context
      helper/LOW falsifier is rejected as a serving substitute. Detail:
      `.work/prefill-helper-c34-20260907.md`; do not confuse it with live text.
- [x] LS-4 Bounded utilization-aware third-slot admission falsifier: the parent health
      predicate was implemented behind `QWEN_ADMIT_UTIL`, but all predeclared 40/60/80 ms
      thresholds damaged the established-four playback envelope despite making the fifth
      request interactive. Keep default-off; do not run a local threshold qualification.
      Cap2/q4 fail-fast remains the control. Detail:
      `.work/ls4-utilization-aware-admission-20260908.md`.

### P4 Overlap and decoder structural cost

- [x] Same-pool decoder consumer tested and rejected: `QWEN_DECODER_THREAD=1` on the
      engine pool caused C4 STREAM_RTF p95 `0.847 -> 1.296`, TTFA p95 `174 -> 1126 ms`
      and max-gap p95 `511 -> 1286 ms`; it observed `group=1` and did not preserve the
      inline decoder batching path. Keep default-off; detail:
      `.work/p4-same-pool-decoder-20260907.md`.
- [x] Fused residual Design-D epilogue passed the CLI byte/audio gate, a short server
      A/B in both per-slot and ragged forms, and a pooled five-minute mixed-bank C4 SOAK:
      short A/B STREAM_RTF p95 `0.831 -> 0.788`; SOAK p95 `0.8933` with zero errors and
      hard p95 `<1` in every window. Promote as an isolated **default-off** candidate;
      per-class p95 remains under-sampled. C5/C6 screens fail startup/safe-start despite
      STREAM p95 <1. Detail:
      `.work/p4-fused-residual-20260907.md`.
- [x] F1 fused-residual × quantum screen (2026-09-08): fused-on q4 is the next C4
      playback/realtime reference candidate (STREAM_RTF p95 `0.868`, prebuffer p95
      `201 ms`, stall@250 `0%`); q8 remains the higher-throughput control and q2 misses
      the preferred STREAM p95 target. Three-wave screen only; not a qualification, and
      no causal fused-vs-off frontier shift was isolated. Detail:
      `.work/f1-fused-quantum-20260908.md`.
- [x] F-cap3 C5 capacity screen (2026-09-08): cap 3 accepted the fifth-request
      wave without the multi-second parent-backlog tail, but cap-3 C5 failed the
      realtime promotion gate (`STREAM_RTF` p95 `0.969`, fifth-launch proxy `1.028`,
      stall@250 `13.3%`). Cap 2/q4 remains the reference; established-four causal
      impact is UNKNOWN because the short run used true simultaneous waves. No C6.
      Detail: `.work/f-cap3-c5-capacity-20260908.md`.
- [ ] DL-1 4+4 intra-CCX decoder lane, default-off `QWEN_SD_LANE_SPLIT=N` (2026-09-09):
      the worker mask is split into a STEP part (engine pool: Talker/CP/prefill) and a
      DECODER part (private pinned team, never the engine pool or its submit lock); the
      frame loop enqueues one bounded decoder unit per slot and blocks only when that
      slot needs another quantum while its unit is in flight (lead <= 1 quantum). Built
      from the single-CCX lane law (`T(B) = 40 + 13.5·B` ms, decoder 9.7 ms per slot,
      Talker+CP saturate the CCX at 2-4 threads) and the L3 contention falsifier (+12 %).
      A/B: one worker on one CCX, 1.7B, fixed text, q4, SL-1, inline vs lane at B2/B3/B4
      (+B5 if B4 is healthy). **GO**: B3 STREAM p95 <= 0.85, B4 <= 0.92, stall@250 = 0,
      no lifecycle/correctness issue; **strong GO**: B4 <= 0.90 without TTFA/prebuffer
      regression; **FAIL**: < 10 % better than inline at B3/B4, or Talker/CP inflation
      erases the overlap, or the mailbox recreates equivalent blocking, or lifecycle is
      unsafe. PASS -> 4x8 host screen at C8/C12/C16; FAIL -> stop, use the measured
      split to decide whether res1/VNNI decoder work is the next lever. Same task:
      `vnni-bf16-product` lane (native bf16 prefill; the f32 pin of `vnni-product` is a
      backend-selection defect) and `QWEN_POOL_SPIN=65536` promoted in the VNNI product
      lanes (measured 2x16 C8 0.893 -> 0.808). **A/B done 2026-09-09: NOT GO, not FAIL** —
      iteration wall matched the prediction (B3 64 ms, B4 72 ms; decoder-call spikes gone,
      stall@250 at B4 100 % -> 0 %) but STREAM p95 B3 0.871 / B4 0.997 miss the gate: the
      4-thread STEP side inflated Talker+CP by +27-29 % (per-slot region sections, ~8 ms per
      slot) and the 2 s clip pays the pipeline's fixed latency (+0.05 STREAM, +30-64 ms
      TTFA). Kept default-off; no host screen. Next lever per the split: the step side
      (5+3 / 6+2 split, long-bank A/B), not res1. **5+3 and 6+2 run 2026-09-09: both
      worse than 4+4 (fixed B4 1.113 / 1.364; long B4 1.015 / 1.263) — the decoder needs
      >= 4 cores to stay hidden at B4 and the step side gains only 3-7 ms from 5-6
      threads; no host screen; 4+4 is the allocation of record, architecture promoted,
      allocation not. DL-2 elastic 8<->4+4 (`QWEN_SD_LANE_ELASTIC=1`, pool width capped only
      while a decoder unit is in flight, preallocated per-slot handoff) run the same day:
      fixed B4 0.987 vs static 0.997, long B4 0.895 vs 0.906, Talker+CP 69.5 vs 69.8 ms —
      the static-partition tax is NOT the cause; the step is slowed ~2x only while the
      decoder unit runs (CP loses L3 residency to the decoder's f32 activations). Next
      lever: the decoder unit's cache footprint, measured by CP ms during overlap.**
      DL-3 falsifiers (2026-09-10): sub-quantum decode, direct ConvT/dwconv/input, NTA
      weight prefetch, hot lane workers, q8 — none moves the CP-in-overlap tax (35-38 ms
      vs 23.5); q8 reaches long B4 0.864 but at prebuffer 806 ms / stall@250 100 %. The tax
      is ~+20 ms per overlapped iteration whatever the decoder does; only the overlap
      share (decoder time on 4 cores, 15-16 ms/frame) scales it. **Next: DL-4 = res1/conv
      kernel efficiency on the lane (fewer weight re-reads, no separate f32 panel),
      metric = decoder unit ms on 4 threads and overlap share.** DL-4 built
      (`QWEN_SD_RES1_V2=1`, direct dilated conv, per-position quant, 4x4 register tile,
      weights read once per time block): res1 1.72x, unit 64 -> 50 ms, overlap share
      48 -> 39 %, lane B4 long 0.869 / fixed 0.918 (gate met); **4x8 host screen: C12
      STREAM p95 0.80-0.81 prebuffer 247 ms stall@250 0 %, C16 0.88 long / 0.92 short
      prebuffer 360 ms stall@250 0 % — twice the inline C8. Screen only: next = SOAK
      C12/C16 with a qualified profile and the V2 numerics ear/mel-qualified.** Detail:
      `.work/dl1-decoder-lane-split-20260909.md`. **Qualification sprint 2026-09-09
      (revision 28d6436, frozen `turin-c8a-32c-vnni-product`, control `-control`):** V2
      quality automated PASS (52 paired files, mel-corr >= 0.9948, ASR CER equal, wav_qc
      equal; ear verdict pending on the Mac listening set); **C12 QUALIFIED for the
      mandatory contract** in every class (waves STREAM p95 0.82-0.85, prebuffer p95
      ~260 ms, safe-start 467 ms, stall@250/@500 0; 30-min SOAK 2205 req 0 errors, pooled
      STREAM p95 0.912, TTFA p95 170, resources/drift PASS) with the preferred 0.90 gate
      missed only by the short (0.959) and conversational (0.914) classes under closed-loop
      soak; Poisson 1.5/2.5 req/s TTFA p95 172/175 ms; overload fail-fast works (per-worker
      cap). **C16 NOT RUN**: the spot host was reclaimed before Phase D. Handoff:
      `.work/turin-vnni-final-handoff-20260909.md`.
- [x] TQ-1 C16 density qualification (2026-09-09, on-demand c8a.8xlarge, revision e1b1ec7):
      waves STREAM p95 0.91-0.96, 30-min soak FAIL (pooled p95 1.004, short 1.045, 596
      per-worker rejects, 111 broken-pipe errors) — C16 = hard-capacity boundary, not a
      product point. Sweep C10-C16 + 10-min soaks C10/C11: knee at C13 (first B4 worker);
      **preferred C11** (pooled soak 0.886; short class alone 0.917, and 0.905 at C10),
      **mandatory-qualified C12**, **hard capacity C16**. Handoff §3.
- [ ] TQ-2 Fail-fast boundary: at a full host rejects surface as TCP resets / broken pipes
      instead of a 503 (4 of 28 in the C12 Poisson run, 111 of 707 in the C16 soak) — the reject path must drain the
      request before closing; also record that rejection is per worker (cap 4): C20 sent 8
      rejects with 16 host slots. Gate: 0 resets over >= 100 rejects, reject count = C-16
      for a simultaneous wave when the parent balances.
- [ ] TQ-8 Leading silence before speech: a measured ~0.5 s of dead air ahead of the first
      voiced frame on a 1.7B-class checkpoint (median 0.50 s over 36 files) against 0.06 s on
      a 0.6B-class one (52 files), consistent across every text class. It is not covered by
      any latency metric we gate on: what a caller experiences is TTFA PLUS the lead-in, so
      ~740 ms against ~186+60 ms. That is larger than anything the C12-WIN decoder ladder was
      chasing, and the ladder delivered nothing. CAUSE NOT ESTABLISHED -- model-emitted silent
      frames or an engine/prompt artefact are both open, and checkpoint size is confounded
      with training data. FIRST STEP is the discriminator, not a fix: run the same
      energy-envelope pass on the OPEN 1.7B and 0.6B models, same bank and settings; ~10
      minutes, CLI is enough. Only if it is model-side does a bounded, default-off leading
      trim make sense, gated on `safe_play_start` rather than TTFA and checked against the
      streaming decoder's continuity contract. Detail:
      `.work/leading-silence-perceived-latency-20260910.md`.
- [ ] TQ-7 GPU serving: `--backend cuda --prefork N` is silently broken. VERIFIED at HEAD:
      the resident CUDA Talker/CP state is created in `main.c` (~:1665) BEFORE
      `qwen_tts_serve_prefork` (~:3082) forks; a CUDA context does not survive `fork()`, and
      no guard exists anywhere (`grep -ci cuda qwen_tts_server.c` = 0, no mutual exclusion in
      main/qwen_tts/cuda). macOS escapes only via the non-Linux prefork stub. Silent wrong
      answer, not a crash. Fix: refuse the combination, or fall back to the single-process
      batched server with a warning. Related: the global GPU seam is bf16-only
      (`qwen_tts_backend.h` exposes only `matvec_bf16`/`matmat_bf16`), so `--backend cuda`
      with `--int8` offloads nothing while the startup line still advertises offload.
      Scoping note: specs 11A/12 and the Arm decoder work carry NO value on a GPU lane, since
      a GPU-resident decoder replaces that component rather than tuning it; the
      backend-agnostic layers do carry over. Detail:
      `.work/arm-linux-v2-parity-track-20260910.md` section 2e.
- [x] TQ-6 BUILD BREAK, not Arm-specific: the tree does not link when neither
      `__ARM_FEATURE_DOTPROD` nor `__AVX512VNNI__` is defined — `SIMD=portable` (the default
      non-VNNI x86 target) and `SIMD=scalar` both fail. Seven symbols are declared and called
      unconditionally but defined only inside the ISA guard in `qwen_tts_kernels.c`, and the
      `#else` fallback sits inside that guard, so it is unreachable. VERIFIED at HEAD with
      `make blas ARCH_FLAGS="-march=armv8-a"`. Partly introduced by C12-WIN: `_ctx` in
      ddfa5d8, `_pack_stack`/`_stack_epilogue` in edfd3fb. FIXED on `feature/arm-parity-vnni`:
      the ISA-neutral ConvT stack and the DL-4 packer moved outside the guard, no-op fallbacks
      for the three ISA-bound entry points, link-only CI jobs. Re-verified with
      `-march=armv8-a` (links, self-test PASS) and on the native build. Detail:
      `.work/arm-linux-v2-parity-track-20260910.md` section 1.
- [x] TQ-5 HTTP JSON string parsing: **ROOT-CAUSED + FIXED** in
      `cf8dd6b09d6de8abc51cccfa6aa90d3fa062b8c7`. The server now decodes standard JSON
      escapes, UTF-16 surrogate pairs, and raw UTF-8 correctly; malformed strings are
      explicit HTTP 400 errors rather than absent optional fields, and JSON responses
      preserve non-ASCII UTF-8. Causal Turin C1 gate passed: escaped/raw requests converge
      to `tail_len=24`, 53 codec frames, and the CLI-identical full codec SHA. The Python
      harness default `json.dumps()` remains the regression oracle; it was not globally
      changed to `ensure_ascii=False`. Requalification is needed for previous non-ASCII
      semantic-quality/CER/golden evidence. Paired V2/control comparative performance
      evidence remains usable; no full C12 performance rerun is required. Detail:
      `.work/server-cli-italian-correctness-20260910.md`.
- [x] TQ-4 Server-vs-CLI Italian pronunciation defect: **ROOT-CAUSED + FIXED** by TQ-5.
      The defect was upstream JSON decoding, not Talker/CP/KV/V2/GEMM, batching, or the
      decoder. The fixed-tree listening pair is retained privately for human sanity review.
      Previous absolute Italian semantic-quality claims remain pending requalification;
      the existing C12-WIN order resumes unchanged after the Spec12/Spec11A gates.
- [ ] TQ-3 Ear verdict on the paired RES1_V2 bank (`samples/tests/2026-09-09_turin-qualification/`);
      PASS promotes `turin-c8a-32c-vnni-product` from provisional to qualified for C12.
- [ ] Reduce structural decoder intercept/rendezvous cost only where measurements justify it;
      retain fused residual as a qualified pooled candidate and consider a strip executor only for proven
      small-call/intercept work. Ragged worker scratch reuse was rejected as a serving
      optimization; claim-first allocation hygiene is retained but KPI-neutral. Details:
      `.work/p4-rag-panel-scratch-20260907.md`, `.work/p4-rag-claim-first-20260907.md`.
- [ ] No speculative completed-stage resumability or dedicated core lanes without evidence
      (DL-1 is the evidence-gated exception: it is an A/B, not a promotion).

### P5 Ownership and batching

- [x] F3 cross-worker cohort coincidence (2026-09-08): in the cap-2 C4 reference,
      useful natural B>=3 opportunities covered only `2.7%` of steady ready events
      within ±1 ms, `3.6%` within ±2 ms and `11.7%` within ±8 ms. Global batching is
      not justified as the next implementation on this 2x6 host; no state consolidation
      or deliberate batch wait was added. Detail:
      `.work/f3-cross-worker-cohort-coincidence-20260908.md`.
- [ ] EO-1/EO-2 Single-engine/global Talker/CP ready set only after P3/P4 coupling is controlled;
      form deadline-compatible cohorts without waiting solely to create B.
- [ ] AMX Talker/CP only when real B >= 4 work exists. CP stateless re-prefill remains dropped
      unless new local evidence invalidates the reviewed cost model.
- [ ] Later ownership/topology changes only if the bounded overlap evidence justifies them.

### Research-only (not current implementation scope)

- [ ] SL-2 live incremental text / park-not-pad; long-form segmentation with decoder-state
      carry; bounded Talker memory; own-codes re-prompt negative arm.

### P6 Qualification and backend comparison

- [x] QL-1 1.7B final decision on GCP C4 highcpu-16: known-text SL-1 removes the
      dominant long-prefill startup term, but full C3 still lacks sufficient sustained
      tail margin. C2/cap2 is the highest full-envelope GOOD point; C3 is screen-only.
      Detail: `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- [x] QL-2a Cross-ISA serving parity audit: common server semantics are portable, but
      AMX Design-D/fused ragged decoder execution is not shared by VNNI or Arm; freeze
      a common-control lane plus a separately labelled best-per-ISA lane before spend.
      Detail: `.work/cross-isa-serving-parity-audit-20260908.md`.
- [x] QL-2b Operational cross-ISA profiles and strict resolved-dispatch gates: AMX,
      VNNI, Arm and common-control profiles pin the relevant flags, reject invalid
      fallbacks and embed the resolved preflight in WAVE/SOAK artifacts. No hardware
      comparison is closed by this task. Detail:
      `.work/cross-isa-operational-parity-20260908.md`.
- [x] QL-2c Local AMD/Turin campaign preparation: known-text SL-1 is pinned across the
  comparable VNNI/Arm/control lanes, the default-off stage-pressure trace has an
  offline receive-gap overlap helper, and the claim audit/runbook preserve
  MEASURED/DERIVED/PREDICTED boundaries. No host was benchmarked. Detail:
  `.work/post-8core-codex-review-20260908.md` and
  `.work/turin-vnni-campaign-plan-20260908.md`.
- [x] QL-2d Turin fast screen on AWS c8a.8xlarge (32 Zen5 cores, 4 CCX, 2026-09-08):
  1.7B holds C8 and not C10 (`2x16` cap 4 STREAM p95 0.79-0.84 at C8, C10 1.05; `4x8`
  cap 2 0.87; `1x32` collapses at 1.4); 0.6B `4x8` cap 4 holds C12 at 250 ms and C16
  at 500 ms, `2x16` collapses at C16. Screen only: provisional profile, 1 wave, short
  texts. Detail: `.work/turin-c8a-32c-fast-screen-20260908.md`.
- [ ] QL-2e Turin ceiling calibration: the doctor's physics ceiling is C28-32 for 1.7B
  where the host delivers 8; measure the three named gaps (wide-pool collapse incl. the
  40 GB/s cross-CCX cache rate, the VNNI decoder term now a ×1.5 GUESS, batch scaling
  past B2) with the stage trace at C8/C10 on `2x16`, then run the pre-registered Phase 4
  on `2x16` cap 4 and `4x8` cap 2 only. Same addendum, §5-6.
- [x] DR-1 Doctor wave plan + ceiling: `wave-plan.json` + `tools/doctor_wave.py`
  (`make doctor-wave`) run the recommended grid from one file; every candidate K gets its
  own measured GEMV roof; section 8 CEILING prints physics / model / floor per shape with
  the measured calibration points of the ISA family. Same addendum, §7.
- [ ] QL-2 Re-evaluate promising backends (0.6B, AVX-512/VNNI hosts, ARM) under the same
  playback-aware harness only after QL-1 has one trusted reference and the QL-2a
  + QL-2b dispatch/quality gates are applied; do not present AMX-only decoder work as
  parity.
  Completed slot: GCP C4 highcpu-16 / 8 physical AMX cores. For 1.7B, `1x8`
  is the best topology and C2 is the final full-envelope point; C3 is screen-only
  and C4 is NOT GOOD. For 0.6B, C3 is the final full-envelope point and C4 is
  non-promoted. The next slot is AMD/Turin VNNI, then Axion/Arm. Detail:
  `.work/ql2-gcp-c4-highcpu16-amx-20260908.md` and
  `.work/gcp-c4-highcpu16-amx-product-capacity-20260908.md`.

### Retained, demoted or deferred (ids kept for addenda; none is a current priority)

- Multi-precision waits behind P0-P3: AMX-2 shared representation, AMX-5 BF16 serving
  policy, AMX-10 W4 feasibility; INT8 is the serving reference.
  PREFILL-Q (calibration-aware prefill quantization) is a deferred research arm behind
  the architecture work; detail: `.work/post-p2-streaming-research-agenda.md` R9.
- Superseded by the envelope: AMX-1, AMX-3, AMX-6, AMX-7, AMX-9 (C4 qualification and
  cross-request decoder aggregation are no longer the next bet; aggregate only for
  isolation/cadence, never for width).
- Controls: P0.1/P0.2, P1.1–P1.4, CTRL-1–CTRL-4; deferred X86-2–X86-8 and LATER-1–4.
- Closed: AMX-4, AMX-5, AMX-8, P3.1, P3.2, P3.6, ragged scheduler review
  (`.work/amx-ragged-scheduler-review-3f7e0df.md`), and the ids below.
- [x] P2.1 Runtime parity — `.work/p2-cross-backend-runtime.md`
- [x] P2.2 CP/Talker region parity — same addendum
- [x] P2.3 Batched-head/budget parity — same addendum
- [x] P2.4 Hot-path allocation fixes — same addendum
- [x] P2.5 One engine-owned budget — same addendum
- [x] P2.6 Pool reentrancy reporting — same addendum
- [x] P3.3a Pool capability parity — `.work/p3-runtime-knob-parity.md`
- [x] P3.4 Decoder capability/policy split — same addendum
- [x] P3.5 Effective AMX/x86 decoder knobs — same addendum

## Qualification gates (provisional, become hard only after MT-1)

| dimension | mandatory | preferred |
|---|---|---|
| correctness | parity PASS; errors = rejects = timeouts = 0 | |
| TTFB / TTFA p95 | measured independently | < 100 ms / < 500 ms (<= 700 ms only for better continuity) |
| STREAM_RTF p95 | < 1 | <= 0.90 (<= 0.85 strong) |
| required_prebuffer p95 | reported | <= 500 ms (<= 250-300 ms strong) |
| safe_play_start p95 | reported | <= ~1 s (<= ~800 ms strong) |
| stall_rate@500ms | -> 0 at the operating point | stall_rate@250ms -> 0 |
| admission / slow client | no induced stall on established streams | |

Never promote q32 for RTF, trade cadence for TTFA, manufacture AMX work, or reopen
BF16/W4 as the P1 fix.

## Evidence

`.work/professional-streaming-architecture.md` (cadence law, AMX accounting, candidates, envelope, historical classification); `.work/p2-checkpoint-20260907.md`, `.work/p1-cadence-truth-20260907.md`,
`.work/p2-sq2-direct-convt-20260907.md`, `.work/p2-sq2-direct-dwconv-20260907.md`, `.work/p2-sq2-direct-input-20260907.md`, `.work/p2-sq2-direct-quant-20260907.md`, `.work/p2-input-length-scaling-20260907.md`, `.work/amx-c4-cross-request-20260907.md`,
`.work/amx-c4-chunk-sweep-20260906.md`, `.work/amx-c4-ragged-threshold-20260906.md`,
`.work/amx-native-epic.md`, `docs/reference-gcp-c4-standard-24.md`, `docs/runtime-map-c8a-c4.md`.
