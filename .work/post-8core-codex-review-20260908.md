# Post-8-core capacity judge — Codex implementation review (2026-09-08)

Task · Independently audit `.work/post-8core-capacity-architecture-judge-20260908.md`
against the serving code and the Sep 7–8 evidence before renting a Turin/VNNI host.
This is a review and campaign-preparation record, not a qualification result.

Review baseline · `1a8332735d96994264bb81f7e81655a36da6b8bc`, branch
`feature/x86-amx-vnni-oss`. The local diagnostic/profile changes made after the review
baseline do not change serving policy; their status is recorded in the campaign plan.

## Executive verdict

The judge is directionally correct, but several quantitative statements are derived
from RTF or from a 12-core diagnostic rather than directly measured on the final
8-core loop. The defensible classification is **MIXED**:

* **Physical pressure is real.** The Talker weight pass reaches the measured memory
  roof for part of an iteration, and worker width/topology changes the available
  bandwidth per serving chain.
* **Engine shape is also real.** One engine loop executes admission, per-slot serial
  work, CP, decoder, output and Talker in order. A decoder call is a complete
  blocking call; pool workers may be parked during loop-owned work and rendezvous.
* **The split is not yet phase-attributed on the 8-core host.** Average
  core-equivalents are not a spare-capacity test, but neither do they prove that a
  partitioned decoder team would win. That remains a Turin falsifier, not a design
  decision.

The next paid host should therefore measure phase pressure and real execution
primitives before any overlap implementation. No AMX retuning, batching rewrite,
admission tuning or scheduler change is justified before that measurement.

## Claim matrix

Statuses distinguish what the judge says from what the repository currently proves.
`DERIVED` and `PREDICTED` values remain labelled; they are not promoted to measured
phase timings.

| judge claim | evidence checked | code mechanism | confidence | status / correction |
|---|---|---|---|---|
| Approximately 12 ms marginal per admitted stream on the 8-core `1x8` loop | 1.7B short-bank `STREAM_RTF` p50 values imply about 37/49/60/74 ms at C1/C2/C3/C4 when multiplied by the 80 ms frame period | one continuous loop adds per-slot sampling/embedding/state work and a decoder contribution | high for this model/topology/q4; not a direct per-phase measurement | **STRONGLY SUPPORTED** as DERIVED. Do not call it a universal per-stream constant or a direct timer result. |
| 1.7B → 0.6B saves only about 7–9 ms per iteration | 8-core C3 derived walls are about 60 ms vs 53 ms; model weight-byte arithmetic predicts about 9 ms from the Talker pass | decoder and much CP/state work are shared by model family; Talker weights are smaller in 0.6B | medium-high | **STRONGLY SUPPORTED** as DERIVED/PLAUSIBLE. A paired C1/C2 phase measurement is still UNKNOWN; 0.6B C5 has a nonlinear extra term. |
| Decoder plus per-slot work is the dominant model-invariant floor | 12-core F-cap3 diagnostic: Talker+CP p50 about 38.7/40.8/40.2 ms at B1/B2/B3, q4 decoder about 59/89/117 ms | `qwen_tts.c` invokes one ragged decoder call after per-slot frame work and waits before the next loop step; VNNI/Arm paths are per-item | high on the 12-core diagnostic; not yet a whole-request 8-core attribution | **STRONGLY SUPPORTED**, but scope it to the measured shapes. Whole 8-core phase share is UNKNOWN. |
| Talker is near DRAM roof for only part of the iteration | Talker-shaped roof and host bandwidth show roughly 1.42 GB of 1.7B INT8 weights in roughly 13–15 ms on the 8-core host; `roof_matvec_int8` is an isolated cold-weight primitive | Talker region streams large weight matrices; later CP/decoder/serial phases have different bottlenecks | high for the Talker pass | **STRONGLY SUPPORTED** with a scope correction: this is not a whole-iteration roof claim, and the local roof helper is not a full `T_step`. |
| Decoder is far below AMX arithmetic peak and is glue/rendezvous/small-op dominated | P2/AR audits and decoder source show im2col/quantization, snake, tails and many small BLAS/conv calls; AMX census proves execution, not wall share | decoder call includes preparation, multiple small operations and synchronization around the tile kernel | medium-high | **STRONGLY SUPPORTED** qualitatively. Exact AMX wall share, DRAM traffic and pool-wait share are UNKNOWN. Any “10–20% of request wall” number is not currently proven. |
| Cadence fails before average CPU saturation | 8-core C4 has about 6.7 core-equivalents while `STREAM_RTF` p95 is about .974; 12-core iteration p95 is much larger than its p50 around decoder calls | serial loop windows and barriers can miss an audio deadline while other workers are not runnable or are waiting | high for “average utilization is unsafe” | **STRONGLY SUPPORTED**. It does not establish exploitable spare silicon. |
| Two independent workers can outperform one wide worker because bursts overlap | measured 1x8 > 2x4 on the 8-core host; historical 2x6 > 1x12 and 2x8 > 1x16 at relevant concurrency | independent prefork loops overlap each worker’s serial windows, at the cost of duplicate weight traffic | high for the measured hosts/configurations | **STRONGLY SUPPORTED**, not “whenever” or a universal scheduler law. Topology remains host- and concurrency-dependent. |
| 1x8 > 2x4 is primarily bandwidth-per-thread, not a universal scheduler truth | per-mask roofs and the 8-core screen agree that 4-thread Emerald Rapids workers do not reach the full roof | each 2x4 worker has a weaker Talker weight stream while 1x8 can use all channels | medium-high | **STRONGLY SUPPORTED** locally. It must not be transferred to Turin without CCD/NUMA and per-mask roofs. |
| Previous decoder-overlap experiments did not falsify a properly partitioned pool | private decoder team created oversubscription; same-pool consumer observed group=1, contended submit ownership and disabled the useful ragged path | `qwen_parallel` has a shared submit/barrier model; the tested alternatives changed ownership or decoder semantics | high as a statement about experiment construction | **STRONGLY SUPPORTED**. A partitioned, non-oversubscribed team remains UNKNOWN; this is not evidence that it will win. |
| A 16-core Turin 2x8 is a meaningful falsifier of the chain-shaped wall | current 8-core result plus old C8a/12-core topology evidence provide a different core count and ISA; the campaign can measure both roofs and phase pressure | 2x8 can test independent chains without cross-worker state migration | high as experiment rationale | **PROVEN** as a falsifiable experiment design; every Turin capacity outcome is still PREDICTED until measured. |

### Additional corrections to the judge

1. The “AMX wall share” statement must remain UNKNOWN. The census proves AMX MAC
   reachability and selected decoder paths; it does not provide a whole-request wall
   denominator. The new `[STAGE]` diagnostic can expose coarse decoder/Talker phase
   pressure, but it is diagnostic and not a kernel wall attribution.
2. `STREAM_RTF × 80 ms` is a useful iteration-wall proxy for this q4/frame contract,
   not a literal timestamp of each engine iteration. It includes the serving path and
   can be affected by phase alignment and client metric behavior.
3. The 0.6B result does not prove a universal model-invariant marginal cost. It is a
   natural experiment showing that the smaller Talker did not buy a second stream on
   this host; the decoder/CP and per-slot terms make that outcome plausible.
4. The claim that average idle cores are exploitable is not established. A future
   partitioned pool must first show that `T_step(B,K_step)` and `D(B,K_decoder)` fit
   the audio deadline with no oversubscription and without moving the memory roof.

## Verified pipeline and ownership map

For `1xN`, the server enters the direct continuous/batched path. For `2xN`, a
prefork parent adds accept/dispatch and fixed-slot behavior; the F2 pre-accept backlog
tail is not present in the direct one-worker path. In either case, within one worker
the current loop is structurally:

```text
admit / inline known-text prefill
  -> install request state
  -> codec head
  -> per-slot sample
  -> batched CP prediction
  -> per-slot embed/VQ/state update
  -> decoder call(s) and PCM callback/write
  -> batched Talker step
  -> next frame/iteration
```

The loop thread owns sequencing and waits for `qwen_parallel` regions. The pool does
not make the loop non-blocking: CP, Talker and decoder regions return only at their
existing boundaries. The decoder call is not preemptible, and synchronous output can
block the same loop on a client write; the bounded async output path exists but is
default-off and passed the relevant slow-reader probe on the 8-core generation.

Current request-local state (KV, CP state, decoder state, text/trailing state and codec
history) is owned by a worker process/slot. No cross-worker migration or global state
consolidation is implemented. This is why F3 rejected global Talker/CP batching on the
12-core host and why Turin must not be benchmarked as if such batching existed.

The new `QWEN_STAGE_TRACE=1` diagnostic emits one line per completed active engine
iteration with monotonic phase totals, active/stepped counts, decoder group size,
ragged/per-item/external classification and synchronous output time. It does **not**
yet join a client receive gap to an engine iteration, does not measure per-phase pool
wait independently, and does not turn the next KPI run into citation-grade evidence.

## Profile/parity blocker resolved locally

The operational `vnni-product`, `arm-product` and `common-control` profiles had
`QWEN_TTS_STREAM_LAYOUT=0` while the current AMX product lane used the validated
known-text SL-1 layout. That would have reintroduced a known startup/prefill confounder
into the non-AMX comparison. The profiles now pin `QWEN_TTS_STREAM_LAYOUT=1` with the
same explicit scope: known text, no live incremental network text, and no implied ICL/
clone resumability. Profile tests assert the pin.

This is a parity/campaign correctness fix, not a claim that VNNI or Arm has AMX
decoder parity. The VNNI lane still resolves `per-item-int8-vnni` as an intentional
`VALID FALLBACK`; Arm resolves its explicit per-item DOTPROD leaf. The common lane
requests decoder batch `0` and is the shape-control lane.

The older prose in `configs/perf/README.md` also said that all x86 profiles pinned
decoder batch to zero. It now describes the current per-profile contract instead:
AMX/VNNI product request batch 1 (with VNNI per-item fallback visible), common-control
batch 0; older machine-specific profiles remain independently qualified artifacts.

## Doctor v2 decision

Doctor v1 remains useful as a fast identity/roof/dispatch/topology predictor, but its
frame-cost model uses transferred constants and an interpolated batch scale. Its own
code documents that it cannot see admission/prefill coupling, and the 8-core C4 result
showed why a predicted `rho` is not a qualification.

I did **not** implement a fake full-TTS Doctor v2. That would require invasive model
loading, request-state setup and a simulator whose numbers could be mistaken for a
serving result. The bounded design for the paid host is:

* use the real model/binary and a short microprobe, not random fake constants;
* measure `T_step(B,K)` for B=1,2,3,4 and K=4,8,16 where the topology supports it,
  using the actual Talker/CP region shape and existing runtime flags;
* measure `D(B,K)` with the actual decoder shapes and resolved ISA leaf, including
  preparation and epilogue, not tile throughput alone;
* record setup, wall, effective B and backend leaf, then use
  `q*T_step + D` only as a **PREDICTED pre-screen** for topology ordering;
* retain WAVE/SOAK as the only qualification path.

`tests/roof_matvec_int8.c` is a cold B=1 Talker GEMV roof helper. `--matmat-bench`
is a shape/capability probe. Neither is Doctor v2 and neither can predict decoder
cadence by itself. If a real-model microprobe cannot be isolated cleanly on Turin,
the campaign should record the design as UNKNOWN rather than fabricate a simulator.

## Physical vs engine decision

Current verdict: **MIXED, with engine-shape evidence strong enough to justify a
diagnostic, but not strong enough to authorize overlap implementation**.

### Accepted before Turin

* phase sequencing and fixed ownership explain why average utilization is not a safe
  admission test;
* Talker memory pressure and per-mask topology are physically relevant;
* decoder preparation/rendezvous and per-slot serial work are real candidates for the
  deadline tail;
* VNNI/Arm must be compared with their actual per-item decoder leaves, not with AMX
  ragged claims;
* `2x8` and `1x16` are the first meaningful Turin topologies, with `4x4` gated by a
  cheap 0.6B chain preflight.

### Not accepted as facts

* a universal 12 ms marginal cost;
* a whole-request AMX wall percentage;
* that partitioned decoder overlap will win;
* that Turin C3/C4/C5/C6 outcomes follow the judge’s numerical predictions;
* that average core-equivalent or vendor bandwidth alone establishes capacity.

## Next action

Use the pre-registered plan in `.work/turin-vnni-campaign-plan-20260908.md`.
It freezes the current profiles and semantics, fingerprints the real AMD topology,
measures per-mask roofs, builds strict VNNI/common-control binaries, screens only
high-information topologies/concurrency points, and enables stage pressure only at
the first failing point. No AWS access or new architecture implementation is needed
until that baseline exists.

Review conclusion · **READY FOR A CONTROLLED TURIN/VNNI CAMPAIGN, with all numerical
capacity claims retained as measured/derived/predicted at their original scope.**
