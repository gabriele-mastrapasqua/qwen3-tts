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

### P2 Small-quantum decoder — detail: `.work/professional-streaming-architecture.md` E3, E4

- [ ] SQ-1 Streaming strip executor: strip → snake → conv1 → snake → conv2 → residual,
      direct INT8 A preparation, packed transposed conv, few rendezvous. CT-2 = GO.
- [ ] SQ-2 Remove fixed per-call work that grows badly as chunks shrink (materialized
      im2col, separate quantization pass, per-tap BLAS scatter, per-call scratch, whole-chunk
      rendezvous); the first direct ragged transposed-conv slice is implemented
      default-off — detail: `.work/p2-sq2-direct-convt-20260907.md`.
- [ ] SQ-3 Report amx_dispatch_share, amx_matrix_mac_share, amx_addressable_mac_share and
      amx_request_wall_share separately after each SQ change; never optimize task count.

### P2 -> P3 gate: post-P2 architecture review — detail: `.work/post-p2-streaming-research-agenda.md`

- [ ] AR-1 BLOCKED on the Codex P2 checkpoint (runtime committed, evidence addendum,
      stable HEAD). Then a read-only architecture review of THAT HEAD: input-length-
      independent first play, incremental/preemptible prefill, bounded-window continuity,
      staged serving (vLLM-Omni mechanisms on CPU), decoupled pipeline, lead as the
      cross-stage currency, long-input qualification; one coherent target architecture
      with falsifiers and a do-not-implement list.
- [ ] AR-2 Freeze the revised P3/P4 ordering from AR-1 before Codex resumes; the P3 tasks
      below are retained but subject to refinement/reordering by AR-1.

### P3 Lead-aware streaming scheduler — detail: `.work/professional-streaming-architecture.md` E2, E7

- [ ] LS-1 Per-stream playback state (delivered audio, lead, time-to-underrun, first-audio
      deadline, admission state) and a design for deadline/slack ordering; not a fixed
      priority ladder.
- [ ] LS-2 Lead-controlled decode/output quantum replacing the static chunk ramp: tiny at
      startup, grows with lead, bounded lead window, no giant bursts.
- [ ] LS-3 Bounded decoder slices that cannot stop codec generation for unrelated streams
      (same pool first; dedicated lane only with evidence).
- [ ] LS-4 Deadline-aware admission: prefill deferred/interleaved near underrun; CT-3
      found a ~309 ms inline prefill and one potentially enlarged overlapping gap.

## Later

### P4 Execution ownership / topology — detail: `.work/professional-streaming-architecture.md` E6

- [ ] EO-1 Clean comparison of prefork/local batching vs single-engine global ready set,
      only after LS-3 controls decode bursts; small mechanism experiments, not a matrix.
- [ ] EO-2 Single-engine Talker/CP batching with bounded decoder slices; CT-4 retained.
- [ ] EO-3 Dynamic core allocation instead of fixed 2x6/2x8; asymmetric lanes only if EO-2
      shows a compute-bound Talker at B >= 3.
- [ ] AMX-0 Trusted execution map with useful AMX wall share (demoted from P0; detail:
      `.work/amx-native-epic.md`).

### P5 Professional output / backpressure

- [ ] OUT-1 Bounded per-stream PCM queue and non-blocking writer; slow/stopped/disconnected
      client cannot stall other streams.
- [ ] OUT-2 Tests: slow reader, stopped reader, disconnect, cancellation/barge-in, queue
      overflow policy.

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

`.work/professional-streaming-architecture.md` (cadence law, AMX accounting, candidates, envelope, historical classification); `.work/p1-cadence-truth-20260907.md`,
`.work/p2-sq2-direct-convt-20260907.md`, `.work/amx-c4-cross-request-20260907.md`,
`.work/amx-c4-chunk-sweep-20260906.md`, `.work/amx-c4-ragged-threshold-20260906.md`,
`.work/amx-native-epic.md`, `docs/reference-gcp-c4-standard-24.md`, `docs/runtime-map-c8a-c4.md`.
