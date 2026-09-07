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
- Best short C4 control (ragged threshold 2, chunk 32): STREAM_RTF p50/p95
  0.900/0.954, TTFA p95 ~565 ms, zero errors, but zero-buffer required prebuffer p95
  ~2.4 s and stall_max ~2.0 s. `STREAM_RTF < 1` is a capacity fact, NOT a continuous
  playback proof; that older interpretation is superseded.
- Prebuffer follows the audio quantum (chunk 8 → 0.45 s, chunk 32 → 1.2-2.5 s) while
  STREAM_RTF barely moves. Chunk 32 is an RTF artifact and is not a production winner.
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

### P1 Cadence truth (current binary, Tier A only)

- [ ] CT-1 Chunk-quantum discriminator at C3 and C4 on 2x6: chunk 8, chunk 32, chunk 8
      with gang join disabled; `[DECODE] dur_ms` and `[ITER]` traces on
      (`QWEN_TTFA_TRACE=1`). Pass = required_prebuffer p95 tracks rho_f x quantum and
      stall_rate@500 separates the arms; also yields CT-2 and CT-4 data.
- [ ] CT-2 Decoder fixed intercept and per-frame slope from dur_ms vs frames, plus
      `[SDPHASE]` attribution (tile vs glue). Intercept < 10 ms and slope < 8 ms/frame
      demotes SQ-1.
- [ ] CT-3 Quantify inline decoder blocking and admission/prefill interference on
      established-stream cadence (new scenario: established streams + new arrival).
- [ ] CT-4 Talker step wall at B=1 vs 2 in one process from `[ITER]`; > 1.6x kills EO-2.
- [ ] CT-5 C2/C3/C4 playback envelope on the current architecture with the MT metrics;
      classify each as GOOD / MARGINAL / NOT STREAMABLE.

### P2 Small-quantum decoder — detail: `.work/professional-streaming-architecture.md` E3, E4

- [ ] SQ-1 Streaming strip executor design note: strip → snake → conv1 → snake → conv2 →
      residual, direct INT8 A preparation, packed transposed conv, few rendezvous; go/no-go
      from CT-2.
- [ ] SQ-2 Remove fixed per-call work that grows badly as chunks shrink (materialized
      im2col, separate quantization pass, per-tap BLAS scatter, per-call scratch, whole-chunk
      rendezvous), bit-parity or mel-corr gated, measured by CT-2 re-run.
- [ ] SQ-3 Report amx_dispatch_share, amx_matrix_mac_share, amx_addressable_mac_share and
      amx_request_wall_share separately after each SQ change; never optimize task count.

### P3 Lead-aware streaming scheduler — detail: `.work/professional-streaming-architecture.md` E2, E7

- [ ] LS-1 Per-stream playback state (delivered audio, lead, time-to-underrun, first-audio
      deadline, admission state) and a design for deadline/slack ordering; not a fixed
      priority ladder.
- [ ] LS-2 Lead-controlled decode/output quantum replacing the static chunk ramp: tiny at
      startup, grows with lead, bounded lead window, no giant bursts.
- [ ] LS-3 Bounded decoder slices that cannot stop codec generation for unrelated streams
      (same pool first; dedicated lane only with evidence).
- [ ] LS-4 Deadline-aware admission: prefill deferred or interleaved when an established
      stream is near underrun; new requests cannot starve; measured by CT-3 scenario.

## Later

### P4 Execution ownership / topology — detail: `.work/professional-streaming-architecture.md` E6

- [ ] EO-1 Clean comparison of prefork/local batching vs single-engine global ready set,
      only after LS-3 controls decode bursts; small mechanism experiments, not a matrix.
- [ ] EO-2 Single-engine Talker/CP batching with bounded decoder slices (gated by CT-4).
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
- [ ] QL-2 Re-evaluate promising backends (0.6B, AVX-512/VNNI hosts, ARM) under the same
      playback-aware harness only after QL-1 has one trusted reference.

### Retained, demoted or deferred (ids kept for addenda; none is a current priority)

- Multi-precision waits behind P0-P3: AMX-2 shared representation, AMX-5 BF16 serving
  policy, AMX-10 W4 feasibility; INT8 is the serving reference.
- Superseded by the envelope: AMX-1, AMX-3, AMX-6, AMX-7, AMX-9 (C4 qualification and
  cross-request decoder aggregation are no longer the next bet; aggregate only for
  isolation/cadence, never for width).
- Controls: P0.1 profiler, P0.2 manifests (`docs/BENCHMARKING.md`), P1.1
  (`.work/config-control-plane.md`), P1.2, P1.3, P1.4 (`.work/p1-4-amx-runtime.md`),
  CTRL-1 (`docs/backend-matrix.md`), CTRL-2, CTRL-3, CTRL-4. Deferred: X86-2, X86-3,
  X86-4, X86-5 (`.work/x86-dataflow-research.md`), X86-6, X86-7, X86-8, LATER-1,
  LATER-2, LATER-3, LATER-4.
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

Never: promote chunk 32 for its RTF; trade cadence for TTFA or RTF for cadence;
manufacture AMX tasks; treat 2x6 as truth; reopen BF16/W4 as the fix; hide negatives.

## Evidence

`.work/professional-streaming-architecture.md` (cadence law, AMX accounting, candidates,
envelope, historical classification); `.work/amx-c4-cross-request-20260907.md`,
`.work/amx-c4-chunk-sweep-20260906.md`, `.work/amx-c4-ragged-threshold-20260906.md`,
`.work/amx-native-epic.md`, `docs/reference-gcp-c4-standard-24.md`, `docs/runtime-map-c8a-c4.md`.
