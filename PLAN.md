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
      admission paths. Local structural smoke and a short GCP 2x6/batch-2 server
      path+census gate pass for short/medium/long text; it remains default-off pending
      ICL/clone, quality and prefill-scaling gates. Detail:
      `.work/sl1-known-text-stream-layout-20260907.md`.
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
- [ ] PF-1 Residual fixed-prompt chunked prefill only where SL-1 leaves a genuinely long
      prefix (ICL/reference or retained non-streaming modes); the existing cloned-context
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
- [ ] Reduce structural decoder intercept/rendezvous cost only where measurements justify it;
      retain fused residual as a qualified pooled candidate and consider a strip executor only for proven
      small-call/intercept work. Ragged worker scratch reuse was rejected as a serving
      optimization; claim-first allocation hygiene is retained but KPI-neutral. Details:
      `.work/p4-rag-panel-scratch-20260907.md`, `.work/p4-rag-claim-first-20260907.md`.
- [ ] No speculative completed-stage resumability or dedicated core lanes without evidence.

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

- [ ] QL-1 Tier B qualification per `.work/professional-streaming-architecture.md` E9;
      discover the maximum GOOD concurrency with margin; publish GOOD streams per cost unit.
      Qualification dimensions must include short/medium/long/mixed inputs and
      long-request arrival scenarios (`.work/post-p2-streaming-research-agenda.md` R8).
- [x] QL-2a Cross-ISA serving parity audit: common server semantics are portable, but
      AMX Design-D/fused ragged decoder execution is not shared by VNNI or Arm; freeze
      a common-control lane plus a separately labelled best-per-ISA lane before spend.
      Detail: `.work/cross-isa-serving-parity-audit-20260908.md`.
- [x] QL-2b Operational cross-ISA profiles and strict resolved-dispatch gates: AMX,
      VNNI, Arm and common-control profiles pin the relevant flags, reject invalid
      fallbacks and embed the resolved preflight in WAVE/SOAK artifacts. No hardware
      comparison is closed by this task. Detail:
      `.work/cross-isa-operational-parity-20260908.md`.
- [ ] QL-2 Re-evaluate promising backends (0.6B, AVX-512/VNNI hosts, ARM) under the same
      playback-aware harness only after QL-1 has one trusted reference and the QL-2a
      + QL-2b dispatch/quality gates are applied; do not present AMX-only decoder work as
      parity.

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
