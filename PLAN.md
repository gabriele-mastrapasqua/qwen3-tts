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
- Current short q8/threshold2 envelope: C2/C3 are GOOD; C4 is MARGINAL
  (`STREAM_RTF` p50/p95 0.793/0.856, required-prebuffer p95 596 ms, stall@500 25%).
  `STREAM_RTF < 1` is capacity, not a continuous playback proof.
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
- Harness (2026-09-07): one metric core `tests/playback_sim.py` with per-request
  safe_play_start, fixed-buffer stall rates, max_gap, coalesced-read share; marks are
  client-observed. Batched server still writes the header with the first audio chunk
  (TTFB = TTFA) via three blocking writes per chunk, no `TCP_NODELAY`: see MT-4.

## Immediate priorities

### P0 Metric truth — detail: `.work/professional-streaming-architecture.md` E1, E8, E11, E12

- [x] MT-1 Receive-mark semantics audited; TTFB stamped independently of TTFA
      (`header_to_audio_ms`); coalesced-read share reported per run — detail:
      `.work/professional-streaming-architecture.md` E12.
- [x] MT-2 Per-request `safe_play_start`, stall_rate/stall_ms @100/250/500/1000 ms,
      max_gap; summaries in the wave and soak analyzers; `tests/test_playback_sim.py`.
- [x] MT-3 Superseded readings corrected in `docs/serving-operations.md` section 5,
      `docs/BENCHMARKING.md` sections 7-8, `ENGINEERING.md` section 9, AWS reference notes.
- [ ] MT-4 Runtime transport fix (no engine change): send the header before synthesis so
      TTFB is a real event, `TCP_NODELAY` or one `writev` per chunk, optional per-chunk
      server flush timestamp trace for a direct server-vs-client mark comparison.

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
      implemented behind `QWEN_SERVER_ASYNC_OUTPUT=1`; C1/C2 Tier-A path, byte-identical
      audio and slow-reader gates pass. Remains default-off pending longer C3/C4 resource
      qualification. Engine enqueue and transport-write timestamps remain distinct.
      Detail: `.work/stream-output-isolation-20260907.md`.
- [x] SL-1 Known-text official dual-track layout implemented behind
      QWEN_TTS_STREAM_LAYOUT=1 and carried through CLI, batch and continuous-server
      admission paths. Local structural smoke and a short GCP 2x6/batch-2 server
      path+census gate pass for short/medium/long text; it remains default-off pending
      ICL/clone, quality and prefill-scaling gates. Detail:
      `.work/sl1-known-text-stream-layout-20260907.md`.
- [ ] LS-1 Credit-gated per-stream lead/deadline state: playable audio lead is the currency;
      never suppress first audio; EDF order only among eligible work.
- [x] LS-3' Small complete decoder calls at safe existing boundaries; q1/q2/q4/q8 floor
      established in a Tier-A C3/C4 screen. q1 is rejected; q2/q4/q8 remain policy
      candidates and no intra-call preemption is claimed. Detail:
      `.work/decoder-quantum-floor-20260907.md`.
- [ ] LS-2 Lead-feedback steady-state quantum: first chunk remains one frame, bounded lead
      window, explicit minimum efficient quantum; q8 remains the upper control until proven.
- [ ] PF-1 Residual fixed-prompt chunked prefill only where SL-1 leaves a genuinely long
      prefix (ICL/reference or retained non-streaming modes); do not confuse it with live text.
- [ ] LS-4 Deadline-aware admission: protect established streams and reject overload rather
      than queue indefinitely.

### P4 Overlap and decoder structural cost

- [ ] Test a same-pool decoder consumer first; promote only if playback and throughput both
      improve. Instrument operation calls, actual pool submissions and pool wait separately.
- [ ] Reduce structural decoder intercept/rendezvous cost only where measurements justify it;
      retain fused residual as quality-gated and consider a strip executor only for proven
      small-call/intercept work.
- [ ] No speculative completed-stage resumability or dedicated core lanes without evidence.

### P5 Ownership and batching

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
- [ ] QL-2 Re-evaluate promising backends (0.6B, AVX-512/VNNI hosts, ARM) under the same
      playback-aware harness only after QL-1 has one trusted reference.

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
