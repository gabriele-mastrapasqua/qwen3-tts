# Current Plan

`ENGINEERING.md` is normative. This file is only the short task queue; detailed
reasoning and reviewed evidence live in the linked `.work/*.md` addenda.

## Product objective

Serve concurrent streaming TTS correctly, with zero errors/rejects/underruns,
`STREAM_RTF p95 < 1` with useful margin, then acceptable TTFA and cost. Do not
trade steady-state playback safety for headline TTFA.

## Active AMX sequence

- [ ] AMX-0 Trusted execution map: reconcile source invariants, effective flags,
      per-kernel census and the real server denominator. Detail:
      `.work/amx-native-epic.md`.
- [ ] AMX-1 GEMV/small-M feasibility: identify real independent matrix work and
      keep irreducible tiny work on the measured fallback.
- [ ] AMX-2 Shared AMX-first representation: decide how persistent weights serve
      VNNI, AMX INT8 and AMX BF16 without locking the backend to one datatype.
- [ ] AMX-3 Dataflow reformulation: implement only a dependency-proven grouping
      that creates useful matrix work; no speculative autoregressive fusion.
      Detail: `.work/amx-c4-cross-request-20260907.md`.
- [x] AMX-4 Decoder AMX INT8 Design D: persistent B packs, direct activation A,
      real decoder shapes and server batching path.
- [x] AMX-5 Decoder AMX BF16: real `TDPBF16PS`, persistent BF16 B packs and
      separate census path; serving policy remains undecided.
- [ ] AMX-6 C4 qualification and scheduler review. Threshold A/B is material
      but still marginal; detail:
      `.work/amx-ragged-scheduler-review-3f7e0df.md`,
      `.work/amx-c4-chunk-sweep-20260906.md`, and
      `.work/amx-c4-ragged-threshold-20260906.md`.
- [ ] AMX-7 Explain the remaining C4 loss with bounded, low-overhead panel,
      batch, pool-wait and starvation/underrun attribution. Threshold `2` is
      the best short control, not a qualification; the cross-request and low-N
      result is in `.work/amx-c4-cross-request-20260907.md` and the prior
      attribution is in `.work/amx-c4-ragged-threshold-20260906.md`.
- [x] AMX-8 Bounded C4 decode-chunk sweep completed for 8/12/16/24/32;
      no mixed-bank candidate has a stable useful margin. Detail:
      `.work/amx-c4-chunk-sweep-20260906.md`.
- [ ] AMX-9 Final 1.7B/0.6B capacity model and sustainable-stream comparison.
- [ ] AMX-10 W4 storage to AMX execution feasibility; no AutoRound runtime yet.

Current checkpoint: Design D is integrated and reaches the engine-owned server
pool. The valid C4 SOAK remains `STREAM_RTF p50/p95 = 0.942/1.035`; threshold
`2` is the best short mixed-bank control at about `0.900/0.954`, still without
the requested margin. The tested low-N M split regressed targeted streaming
and was rejected; safe cross-request fusion already exists within each worker,
while crossing prefork workers requires a new scheduler/ownership design. The
default remains 8, AMX-6/7 remain open, and C5/C6 are deferred.

## Controls and backend parity

- [ ] P0.1 Runtime profiler with execution tree, pool occupancy, memory traffic
      and resolved backend; detail: `.work/amx-native-epic.md`.
- [ ] P0.2 Canonical benchmark manifests/runbook and clean-commit qualification;
      see `docs/BENCHMARKING.md`.
- [ ] P1.1 Server configuration control plane; detail:
      `.work/config-control-plane.md`.
- [ ] P1.2 ARM i8mm shape/region validation when an ARM host is available.
- [ ] P1.3 VNNI GEMV/GEMM shape coverage and q4 weak-shape follow-up.
- [ ] P1.4 Keep the AMX B>=4 runtime evidence current; detail:
      `.work/p1-4-amx-runtime.md`.
- [ ] CTRL-1 Common-path parity matrix: implemented, selectable and actually
      effective for each backend; see `docs/backend-matrix.md`.
- [ ] CTRL-2 Feature-flag parity and effective/default semantics across backends.
- [ ] CTRL-3 Cloud A/B only for unresolved topology or pool differences.
- [ ] CTRL-4 Module boundaries and one capability/fallback table.

### Closed control-plane work

- [x] P2.1 Common runtime parity and region ownership; detail:
      `.work/p2-cross-backend-runtime.md`.
- [x] P2.2 CP/Talker region and head parity.
- [x] P2.3 Batched-head and execution-budget parity.
- [x] P2.4 Hot-path allocation and pool-interface fixes.
- [x] P2.5 One engine-owned execution budget.
- [x] P2.6 Truthful pool reentrancy/ownership reporting.
- [x] P3.1 Authoritative runtime-knob inventory; detail:
      `.work/p3-runtime-knob-parity.md`.
- [x] P3.2 Backend runtime-knob mapping.
- [x] P3.3a Pool capability parity.
- [x] P3.4 Decoder capability/policy split.
- [x] P3.5 Effective AMX/x86 decoder knobs and region observability.
- [x] P3.6 Decoder-pool configuration now has explicit values, resolved startup
      reporting and fail-fast unknown-value handling; detail is preserved in
      `.work/amx-c4-chunk-sweep-20260906.md`.

## Deferred follow-ups

- [ ] X86-2 Revisit low-B AMX/VNNI crossover only after the serving dataflow
      presents enough useful work.
- [ ] X86-3 Keep persistent packed-RHS consumption opt-in until serving cost is
      requalified with the current dataflow.
- [ ] X86-4 Remove remaining decoder activation gather/pack/scatter overhead.
- [ ] X86-5 Revisit duplicated AMX activation packing only when useful per-worker
      matrix width is established.
- [ ] X86-6 Qualify the 0.6B server on the same C1/C2/C4 methodology.
- [ ] X86-7 Compare 0.6B and 1.7B shape-family and serving shifts.
- [ ] X86-8 Define latency-first and streaming-safe operating profiles.
- [ ] LATER-1 Qualify INT8 prefill separately from the production path.
- [ ] LATER-2 Decide the two unreviewed GCP reference notes and stray local object.
- [ ] LATER-3 Audit BLAS replacement only after structural parity is closed.
- [ ] LATER-4 Rename legacy CLI batch aliases without changing behavior.

## Closed checkpoints

- Backend matrix, effective-config reporting and the major x86/ARM parity audit
  are landed; remaining ARM items are hardware-blocked.
- Decoder Design D and real AMX BF16 are implemented with parity/census evidence.
- Ragged-panel scheduler review is safe to qualify; low findings and targeted
  tests remain explicitly deferred in the linked addendum.
