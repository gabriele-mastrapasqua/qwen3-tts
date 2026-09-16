# Task · Question · Known facts · Unknowns · Files/functions inspected · Evidence · Conclusion · Next action

## Task

Diagnose the Arm v2 sustained closed-loop playback regression before any further
concurrency headline or profile promotion.

## Question

Why do true-wave/parallel runs look materially healthier than continuously replaced
closed-loop streams, and can moving admission prefill to the existing helper path protect
playback continuity?

## Known facts

- The C6/C7/C8 closed-loop mini-soaks reproduce the regression within roughly 2–5 minutes;
  a 30-minute soak is not required for fast iteration.
- The runs complete without crashes, request rejects, request timeouts, or other obvious
  functional failures, but STREAM_RTF tails approach or exceed one, safe-play-start rises,
  and stall@250/stall@500 become material.
- The earlier diagnostic trace showed stable memory/thread/fd/scratch behaviour and no
  accumulating resource signature. This is currently a serving/scheduling/QoS problem,
  not a proven allocator leak.

## Unknowns

- The current closed-loop CSV does not carry a shared monotonic receive clock, so STAGE
  overlap with each individual playback gap is evidence of temporal pressure, not a
  complete causal join.
- The existing helper path is not resource-isolated: it submits work through the same
  engine pool and therefore is not a clean test of a separate prefill CPU group.
- Cohort desynchronization may be a secondary amplifier; it is not yet proven to be the
  primary cause.

## Files/functions inspected

- `qwen_tts.c`: inline admission/prefill loop, `QWEN_PREFILL_SLICE`,
  `QWEN_PREFILL_HELPER`, `prefill_helper_main`, and frame-loop scheduling.
- `qwen_tts_server.c`: fail-fast admission and per-worker slot/queue policy.
- `tests/serve_soak.py`: closed-loop windows, playback metrics and resource sampling.
- `tools/stage_pressure.py`: STAGE phase summary; concurrent diagnostic markers can
  interleave, so the raw logs remain authoritative and summaries use only valid records.

## Evidence

All-on Arm v2, 1.7B OSS/Ryan, 4x8, C8, same profile and request bank, four-minute
closed-loop runs. Both arms used STAGE/LIFE/scratch diagnostics; absolute values are
diagnostic, not production qualification numbers.

| arm | STREAM p50/p95 | TTFA p50/p95 | safe-start p50/p95 | stall@250 | stall@500 | completed | errors/rejects/timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| inline control (`QWEN_PREFILL_HELPER=0`) | 0.934/1.041 | 180.5/214.0 ms | 450.1/716.8 ms | 20.1% | 2.8% | 168 | 0/0/0 |
| helper (`QWEN_PREFILL_HELPER=1`) | 0.986/1.098 | 245.0/308.6 ms | 566.8/1129.3 ms | 46.8% | 10.1% | 163 | 0/0/0 |

The helper arm is worse by +0.057 STREAM_RTF p95, +94.6 ms TTFA p95, +412.5 ms
safe-start p95, +26.6 percentage points stall@250 and +7.3 points stall@500. Its
per-window stall@250 values were 30.4%, 50.0% and 63.8%; this is a treatment-specific
queue/contended-pool regression, not evidence that the baseline has a growing leak.

In the control STAGE records, the long samples were about 299 ms wall with roughly
169 ms admission/prefill. In the helper arm admission/prefill disappeared from the
frame-loop STAGE records, but long samples remained and were dominated by queue/serial
wait (median about 163 ms; p95 about 891 ms), with higher CP/talker tails. The helper
also raised the steady worker thread count from 216 to 220. Ragged/per-item decoder
fallbacks remained zero in both arms; decode time stayed negligible in the problematic
records.

## Conclusion

- True-wave/parallel capacity, sustained closed-loop capacity and realistic arrival-load
  capacity (for example Poisson) are distinct benchmark families. A true-wave result is
  never a sustained qualification.
- The admission/prefill-interference hypothesis is **PARTIALLY CONFIRMED** at the broad
  resource-interference level: new work competes with playback-critical work, and simply
  moving it to a helper thread does not protect playback.
- `QWEN_PREFILL_HELPER=1` is **REJECTED** as the treatment and remains default-off. It
  removes inline prefill from STAGE but worsens playback, showing that a shared-pool
  helper is not QoS isolation.
- The strongest next discriminator is a playback-first admission guard that keeps a new
  request READY/WAITING while active streams are near their next playback deadline.

## P0-A playback-first admission guard

The first reversible guard was implemented in the continuous batched admission path. It
is enabled only by `QWEN_ADMISSION_GUARD=1`; `QWEN_ADMISSION_GUARD_TARGET_MS=400` was
used for this experiment. The guard examines the active streams' server-side ready-audio
lead before taking the next queued job. If the minimum lead is at or below the target,
the queue head remains READY/WAITING and the active playback path continues. No helper
thread, prefill slicing, affinity change, pool change or other architecture change was
combined with this A/B. The ready-audio lead is a scheduling proxy, not client playback
feedback; the experiment used synchronous output. The separate `JOB_SINGLE` queue is
outside this first guard scope.

The control and treatment used the same Arm v2 all-on profile, 1.7B OSS/Ryan model,
request bank, C8 closed-loop load, warmup and four-minute measurement window. These are
diagnostic mini-soak comparisons, not a sustained qualification.

| arm | STREAM p50/p95 | TTFA p50/p95 | safe-start p50/p95 | stall@250 | stall@500 | completed / throughput | errors/rejects/timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| control, guard off | 0.953/1.061 | 185/241 ms | 463/776 ms | 25.5% | 4.7% | 173 / 0.666 req/s | 0/0/0 |
| P0-A guard, target 400 ms | 0.927/1.053 | 320/2805 ms | 710/3066 ms | 13.8% | 6.5% | 147 / 0.572 req/s | 0/0/0 |

The guard reduced `stall@250` by 11.7 percentage points (45.8% relative), but did not
reduce the `stall@500` tail, left STREAM_RTF p95 above one, increased safe-start p95 by
2.29 seconds and TTFA p95 by 2.56 seconds, and reduced completed-request throughput by
about 14%. Resource diagnostics remained stable: 216 threads, 35–42 FDs, and no positive
RSS growth signature in either arm.

The guard counters explain the trade-off. Across the worker reports there were 111
deferred admissions and 87 immediate admissions. Per-worker defer p50 was about
1.03–1.57 s, p95 about 2.78–2.95 s, and the largest defer was about 19.4 s. Deferred
decisions usually saw one active stream (maximum two), pending depth one (maximum two),
and negative ready-audio slack roughly −368 to −402 ms at the typical point, with a
minimum observed slack around −1.56 s. This confirms that the guard fires on real
playback pressure, but the fixed target creates excessive admission starvation.

STAGE timing does not show a new decoder/talker slowdown: aggregate STAGE `wall_ms`
p95 was 111 ms in control versus 106 ms with the guard, and `talker_ms` p95 was 77 ms
versus 75 ms. The guard reduced very long STAGE records, while the visible regression
shifted into admission waiting and fresh-request startup. This strengthens the
admission/QoS interpretation but rejects this particular target as a production policy.

## P0-A classification

**PARTIALLY CONFIRMED.** The experiment proves that delaying fresh admission can reduce
short playback gaps, but the current guard is too coarse/aggressive: safe-start and
new-request TTFA tails become much worse, throughput falls, and the 500 ms playback tail
does not improve. It is therefore not a clean success criterion and is not promoted.
Do not begin cooperative slicing from this result. The next P0 action is a smaller,
bounded admission-policy follow-up using the same C8 mini-soak, with explicit limits on
defer duration and a fresh-admission escape path; only a treatment that improves both
playback continuity and aggregate QoS should advance to slicing.

## P0-B token-range slicing probe

The existing reversible token-range admission path was exercised without the boolean
guard, using the same all-on C8 profile and short closed-loop reproducer. Both treatments
emitted `[ADMSLICE]`, so the result is not a flag-vacuous run.

| slice setting | nonzero prefill p50/p95 | STREAM p95 | safe-start p95 | stall@250 | stall@500 | TTFA p95 | completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 (nominal ~10 ms) | 83.5/96.6 ms | 1.101 | 1361 ms | 19.7% | 7.9% | 220 ms | 84 |
| 48 (nominal ~20 ms) | 82.4/96.7 ms | 1.066 | 1326 ms | 25.3% | 7.6% | 220 ms | 87 |

The nonzero slice occupancy is effectively invariant: both settings hit an approximately
83–97 ms p50/p95 floor. The C8 request shape also showed a short prepared range (for
example, sequence length 10 with prefix length 9), so increasing the token count cannot
preempt the one-token range more finely. The token-range mechanism is therefore
**REJECTED for playback protection in this configuration**; no 72-token run is warranted.

This does not prove that every internal prefill operation is atomic, but it proves that
the current token boundary is above the relevant scheduler granularity. The next narrow
action is component timing inside the range: setup/embedding, per-layer blocks and
KV/finalization. If one layer/kernel owns the floor, use finer-grained preemption only if
it can be made safely; otherwise move directly to genuine resource isolation rather than
reusing the shared-pool helper.

## Prefill floor localization

The short C8 reproducer ran with the existing token-range path and `QWEN_TTFA_TRACE=1`.
It emitted 60 range summaries and 1,661 per-layer records. The measured nonzero range
timing was:

| component | p50 / p95 / max |
|---|---:|
| setup | ~0 / ~0 / ~0 ms |
| one layer total | 2.92 / 3.29 / 7.00 ms |
| full 28-layer range | 83.24 / 88.43 / 101.4 ms |
| finalize | ~0 / ~0 / ~0 ms |

Per-layer phase medians were approximately QKV 0.53 ms, O-projection 0.26 ms,
gate-up 1.37 ms, activation 0.01 ms and down 0.74 ms; rope/KV, norms and attention
were each near zero at this scale. No individual phase or kernel owns the 80–100 ms
floor. The scheduler regains control only after `qwen_talker_prefill_range` has completed
all 28 layers, so the token boundary is above the required QoS granularity.

## Next action

1. P0-B finer checkpoint is complete as a discriminator. With token slicing, guard and
   helper off, the same new binary produced this C8 short-sweep:

   | layer group | nonzero prefill p50/p95 | TTFA p95 | safe-start p95 | STREAM p95 | stall@250 | stall@500 | completed |
   |---:|---:|---:|---:|---:|---:|---:|---:|
   | 0 | 84.9/119.5 ms | 223 ms | 846 ms | 1.049 | 30.4% | 6.3% | 79 |
   | 1 | ~3 ms sampled | 2460 ms | 3380 ms | 1.105 | 25.8% | 11.3% | 62 |
   | 2 | 5.9/10.6 ms | 1208 ms | 2079 ms | 1.076 | 11.8% | 2.9% | 68 |
   | 4 | 11.7/20.1 ms | 757 ms | 1872 ms | 1.131 | 22.9% | 11.4% | 70 |

   Every run had zero errors, rejects and timeouts. **PARTIALLY CONFIRMED**: the
   checkpoint reduces the real occupancy to layer granularity and layer=2 materially
   improves playback tails, but the single pending admission remains head-of-line blocked
   and fresh-request TTFA is not acceptable. No layer size is promoted and no 3/5/6 sweep
   is justified.
2. The resource-isolation discriminator is now complete below, and the topology result
   confirms strong cross-worker bandwidth contention. The next discriminator is a small
   topology-aware serving-width comparison (for example 2x16 versus 4x8), not another
   shared-pool helper, CPU-mask micro-sweep or layer-count sweep. Keep the checkpoint
   behind its flag.
3. Keep the P0-A guard bounded/reversible; do not promote its current 400 ms policy.
4. Only after the scheduling/resource experiment inspect allocation/churn; current evidence
   does not justify a broad malloc/thread refactor.

## P1 resource-isolation A/B

The smallest genuine isolation treatment used the same Graviton5 Arm-v2 all-on C8/4x8
profile and layer=2 checkpoint as the control. It enabled `QWEN_PREFILL_HELPER=1` only
as the admission carrier, `QWEN_PREFILL_ISOLATE_CPUS=1`, and did not use the shared engine
pool for isolated parallel regions. Each prefork worker started with eight CPUs; one was
reserved for admission and the seven-CPU engine mask was passed to lane/decoder setup:

| worker | configured mask | playback/engine mask | admission CPU | helper bind observed |
|---:|---|---|---:|---|
| 0 | 0-7 | 0-6 | 7 | 7 |
| 1 | 8-15 | 8-14 | 15 | 15 |
| 2 | 16-23 | 16-22 | 23 | 23 |
| 3 | 24-31 | 24-30 | 31 | 31 |

The server reported `OpenBLAS threads now 1`, `openblas_threads=1` and
`blas_owned=1`; the worker logs showed the isolated helper bound to each reserved CPU.
Thus this was a real CPU-mask/binding experiment, not the rejected helper that merely
submitted into the shared engine pool. The process stayed at 216 threads, 40--42 FDs,
and RSS did not grow; CPU utilization was not sampled per-core, so no stronger claim of
core balance is made.

The 2-layer trace also exposed the cost of the one-CPU admission partition. Early groups
were about 7.1--13.8 ms, but under sustained load the common group range was 31.2--46.9
ms. A full 14-group prefill therefore measured roughly 425--580 ms of layer work, with
`OpenBLAS=1`; the isolated `qwen_parallel` path is intentionally serial for this
experiment. This is direct evidence that one reserved CPU is underprovisioned for
admission, not evidence of a leak.

The valid control and treatment were short diagnostic runs (one 60-second measured
window, so neither is a qualification soak):

| arm | TTFA p50/p95 | safe-start p50/p95 | STREAM p50/p95 | stall@250 | stall@500 | completed | errors/rejects/timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| layer=2 control | 1110/1392 ms | 1362/2602 ms | 0.958/1.070 | 31.4% | 8.6% | 51 | 0/0/0 |
| isolate=1 CPU | 606/779 ms | 910/3626 ms | 0.976/1.156 | 30.3% | 24.4% | 49 | 0/0/0 |

The treatment improved TTFA p95 by about 613 ms, but it did not retain the layer=2
continuity benefit: `stall@250` was essentially unchanged, `stall@500` worsened by
15.8 percentage points, safe-start p95 worsened by about 1.0 s, and STREAM p95 stayed
above one and worsened by 0.086. The result is therefore:

**REJECTED for this 1-CPU partition.** Isolation and binding were verified, but the
treatment did not protect playback. Since continuity did not improve strongly, the
conditional 2-CPU follow-up is not warranted by this A/B; do not claim that every wider
partition is impossible, only that this underprovisioned partition is not a production
policy. The remaining limit is resource capacity/interference plus admission policy, not
an unproven allocator leak.

## Graviton5 topology/bandwidth discriminator

Before changing admission policy again, the exact 32-core Graviton5 guest was measured
with the existing project tools. Linux exposes 32 physical Neoverse-V3 cores, one socket,
one NUMA node and one 48 MiB L3 (`shared_cpu_list=0-31` for every CPU). L1/L2 are private
per core (64 KiB/2 MiB); no additional cluster/die/cache topology was exposed by the
guest. This is materially different from the four independent 8-core/LLC domains used by
the Turin reference.

The DRAM-oriented `tests/membw.c` curve was run on the full mask with a 384 MiB working
set (128 MiB per buffer):

| mask | threads | Copy GB/s | Triad GB/s | Read GB/s |
|---|---:|---:|---:|---:|
| 0-31 | 1 | 50.9 | 51.4 | 13.1 |
| 0-31 | 4 | 138.3 | 130.2 | 52.0 |
| 0-31 | 8 | 168.2 | 154.3 | 102.3 |
| 0-31 | 16 | 183.4 | 167.8 | 159.0 |
| 0-31 | 32 | 184.6 | 154.9 | 160.8 |

The 8-core and 16-core reference masks reached read roofs of 102.2 and 163.2 GB/s;
these are mask-specific measurements, not host bandwidth divided by worker count.

The existing `tests/roof_matvec_int8.c` was then run as simultaneous processes, with each
process pinned to its own mask. `FRAME(all4)` is 1.344 GB of distinct INT8 Talker weights
in real layer order; each row below is the best of five repetitions:

| layout | worker/frame ms | worker GB/s | aggregate GB/s | per-worker slowdown vs 1x8 |
|---|---|---:|---:|---:|
| 1x4, mask 0-3 | 13.24 | 106.4 | 106.4 | — |
| 1x8, mask 0-7 | 8.82 | 159.9 | 159.9 | 1.00x |
| 2x8, masks 0-7/8-15 | 19.82--22.19 | 63.5--71.1 | ~134.6 | ~2.38x |
| 4x8 contiguous | 38.56--40.10 | 35.1--36.5 | ~143.4 | ~4.45x |
| 4x8 interleaved | 36.25--37.48 | 37.6--38.9 | ~152.7 | ~4.19x |

Thus the decisive quantity is not the isolated 1x8 result but the same worker under
concurrent load: its frame latency rises from 8.82 ms to about 39.3 ms in contiguous
4x8, while aggregate throughput reaches only about 0.90x the isolated 1x8 bandwidth
instead of 4x. Interleaving improves the 4x8 aggregate by only about 6.5%, so it does
not reveal four useful hidden locality islands. No engine-level contention test was
needed after this kernel-level result.

Against Turin's measured ~55.2 GB/s per 8-core CCX and ~208.3 GB/s at 4x8, Graviton5's
isolated 8-core GEMV is strong, but its four-worker aggregate is materially non-scaling.
The verdict is **CONFIRMED for cross-worker shared-cache/memory/fabric contention** and
**not evidence of four independent 8-core bandwidth domains**. This does not prove the
exact physical fabric topology beyond what Linux exposes, but it is enough to explain why
the 4x8 serving partition can amplify sustained interference on Graviton5.

The first discriminator is complete. Do not run another admission-policy soak yet; the
topology-aware serving-width follow-up is recorded below. Raw evidence is private under
`private/arm-topology-discriminator-20260913/`.

## Graviton5 topology-aware serving-width sweep

The requested follow-up reused the same `roof_matvec_int8` binary, 28 distinct layers,
1.344 GB `FRAME(all4)` and five repetitions. The isolated references needed for a fair
per-worker slowdown comparison were also measured:

| isolated shape | frame ms | GB/s |
|---|---:|---:|
| 1x6 | 10.50 | 134.2 |
| 1x8 | 8.82 | 159.9 |
| 1x16 | 10.52 | 134.0 |
| 1x32 | 9.92 | 142.0 |

Simultaneous layouts were:

| layout | worker frame ms | per-worker GB/s | aggregate GB/s | slowdown vs matching isolated worker |
|---|---:|---:|---:|---:|
| 2x16, 0-15 / 16-31 | 17.25--19.07 | 73.9--81.7 | **155.6** | ~1.73x |
| 4x8 contiguous, 0-7 / 8-15 / 16-23 / 24-31 | 38.56--40.10 | 35.1--36.5 | 143.4 | ~4.45x vs 1x8 |
| 4x8 interleaved/spread | 36.25--37.48 | 37.6--38.9 | 152.7 | ~4.19x vs 1x8 |
| 4x6, 0-5 / 8-13 / 16-21 / 24-29 | 35.45--37.06 | 38.0--39.8 | **155.4** | ~3.46x |

The interleaved 4x8 result is only about 6.5% better than contiguous 4x8, so the
affinity comparison does not expose useful hidden locality islands. `2x16` and `4x6`
both improve aggregate GEMV throughput by about 8.5% over contiguous 4x8; `1x32`
minimizes single-worker latency but is not a concurrent serving shape. The winner for
the next serving baseline is therefore **2x16**, with **4x6** a close fallback that leaves
eight CPUs spare. No separate Talker/CP server timing was run: the kernel discriminator
already showed strong contention, so an engine test would not change this topology
decision and no production soak was started.

This confirms that Graviton5 wants a different worker width from the Turin 4x8 layout,
but it does not eliminate shared-fabric contention: even 2x16 slows each 16-core worker
about 1.7x versus its isolated reference. The next experiment is a short C6/C8
closed-loop check using 2x16 as the serving baseline; do not tune admission policy or
claim sustained capacity until that shape is checked. Raw outputs remain private under
`private/arm-topology-discriminator-20260913/gemv-shapes/`.

## Axion cross-host control

The same discriminator was run on the existing 32-core GCP Axion box, using the same
`roof_matvec_int8` source, 28 distinct layers, 1.344 GB `FRAME(all4)`, five repetitions
and contiguous masks `0-7`, `8-15`, `16-23`, `24-31`. Axion exposes one socket, one NUMA
node and one 80 MiB L3 shared by `0-31`; `numactl` was not installed, so node topology
was confirmed through sysfs. It is a Neoverse-V2 guest, whereas Graviton5 is Neoverse-V3.

| layout | worker frame ms | per-worker GB/s | aggregate GB/s | slowdown vs isolated 1x8 |
|---|---:|---:|---:|---:|
| 1x8 | 8.98 | 156.9 | 156.9 | 1.00x |
| 2x8, 0-7 / 8-15 | 10.22--10.23 | 137.8--137.9 | **275.7** | ~1.14x |
| 4x8, 0-7 / 8-15 / 16-23 / 24-31 | 15.97--16.49 | 85.5--88.3 | **347.8** | ~1.81x |

Axion's isolated 1x8 result is within 2% of Graviton5's 159.9 GB/s, so the difference
is not an isolated-kernel advantage. Under four simultaneous workers Axion reaches about
2.22x its isolated aggregate and keeps per-worker latency near 16.2 ms; Graviton5 reaches
only about 0.90x its isolated aggregate and rises to about 39.3 ms per worker. The two
guests therefore expose the same coarse Linux topology but have fundamentally different
cross-worker scaling. The verdict is **GRAVITON5-SPECIFIC CONTENTION STRONGLY CONFIRMED**:
the G5 shared-cache/memory/fabric behavior, not merely the 4x8 software partition or the
admission policy, is a primary contributor to the Arm sustained regression. No Axion
engine soak was run. Raw control evidence is private under
`private/arm-axion-topology-discriminator-20260913/`.

## Graviton4 cross-host control

The requested AWS spot control was run from a clean HTTPS clone of the public feature
branch, with the existing `roof_matvec_int8` target and the same 28-layer, 1.344 GB
`FRAME(all4)` workload. The guest reports 32 Neoverse-V2 cores, one socket, one NUMA
node and one 36 MiB L3 shared by `0-31`.

| layout | worker frame ms | per-worker GB/s | aggregate GB/s | slowdown vs isolated 1x8 |
|---|---:|---:|---:|---:|
| 1x8 | 8.26 | 170.7 | 170.7 | 1.00x |
| 2x8, 0-7 / 8-15 | 9.65--10.07 | 139.9--146.1 | **286.0** | ~1.19x |
| 4x8, 0-7 / 8-15 / 16-23 / 24-31 | 12.91--12.99 | 108.5--109.2 | **435.1** | ~1.57x |

This is much closer to Axion than Graviton5 despite the same coarse one-L3 topology being
visible in Linux. At 4x8, Graviton4 retains about 2.55x its isolated aggregate bandwidth
and keeps each worker near 13 ms; Axion retains about 2.22x and ~16.2 ms, while Graviton5
retains only ~0.90x and rises to ~39.3 ms. The verdict is **Graviton5-specific contention
strongly confirmed by two ARM controls**: the G4 and Axion results rule out a generic
Arm/KAI or generic 4x8 software limitation. No engine soak or spread sweep was run on G4
because the 4x8 discriminator was already decisive. Raw evidence is private under
`private/arm-graviton4-topology-discriminator-20260913/`.

## Graviton4 two-minute closed-loop screen

After the kernel comparison, the public-branch binary passed `--self-test` with zero
failures. The existing Axion Arm-v2 all-on environment was applied explicitly with
`--no-profile`; no profile or hardware qualification was claimed for Graviton4. Runs
used the unchanged 4x8 server shape, Ryan/English, the standard text bank, 15 seconds
warm-up and a two-minute closed-loop window:

| model / point | completed | TTFA p95 | STREAM RTF p95 | safe-start p95 | stall@250 | stall@500 | errors/rejects/timeouts | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1.7B / C8 | 131 | 127 ms | **0.680** | 230 ms | **0%** | **0%** | 0/0/0 | PASS |
| 0.6B / C16 | 128 | 279 ms | **1.084** | 1876 ms | **46.8%** | **14.7%** | 0/0/0 | FAIL playback |

The 1.7B C8 point is clean in this short screen: resource stability passed, max-gap p95
was 329 ms and no 250/500 ms stalls occurred. The 0.6B C16 point did not crash or reject
requests, and resources were stable, but playback was not realtime: pooled prebuffer p95
was 1.745 s and the per-class KPI failed for long/medium. This is a short diagnostic
screen, not a sustained qualification and not evidence that the smaller model's maximum
capacity is C16; it only says C16 is unsafe under this exact 4x8/all-on run. Raw output
is private under `private/arm-graviton4-mini-soak-20260913/`.

## Decoder-lane bounded-async and urgency ordering

On the scalable 32-core Arm control, a C10 diagnostic localized the playback knee to
workers with three active streams. The original synchronous bounded mailbox path made the
central frame loop wait for a busy decoder slot. `QWEN_SD_LANE_NONBLOCK=1` proved that
removing that wait improves STREAM tails but permits an unbounded ready-code backlog.
`QWEN_SD_LANE_NONBLOCK=2` bounds that backlog to one ready quantum behind the in-flight
unit; it removes synchronous `dec_wait_ms` but does not yet make C10 healthy.

The first follow-up, `QWEN_SD_SCHED=urgency`, was deliberately narrow: bounded async,
decoder targets, batch width, admission and cohort policy were unchanged. The new mode
only ranks runnable slots for decoder enqueue by a documented proxy (mailbox-pause age,
then ready decoder backlog, then first-audio state). The sink does not presently expose a
numeric audio lead at this layer, so this is not claimed to be a true lead/deadline policy.

| arm | STREAM p50/p95 | TTFA p95 | safe-start p95 | stall@250 | stall@500 | completed | errors/rejects/timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| bounded async reference | 0.587/0.934 | 211 ms | 895 ms | 7.7% | 0.9% | 127 | 0/0/0 |
| urgency enqueue order | 0.566/1.077 | 335 ms | 1234 ms | 14.0% | 6.1% | 124 | 0/0/0 |

**REJECTED.** The ordering-only treatment does not improve continuity and is worse in the
single comparable measured window. It is not a capacity or qualification result.

The sampled slot trace is more useful than the absolute A/B: all observed paused slots
resume, so there is no evidence of a permanently lost stream; however, individual pause
runs reach roughly 170 central-loop turns. Decoder completion groups are predominantly
singletons (5,316 singleton groups versus 366 two-slot groups in this diagnostic), so
cohort alignment remains a credible secondary issue. The next minimal discriminator is
not another priority weight: add lifecycle timestamps for enqueue, lane start, completion
and central-loop resume, then determine whether the residual tail is queue/inflight time
or completion-to-wake delay. Raw trace remains private.

## Decoder lifecycle timing — active diagnostic

**Hypothesis.** The residual C10 tail after bounded async is localized in one of five
intervals: ready-to-enqueue, enqueue-to-lane-start, lane compute, lane-done-to-main-loop
observation, or resume-to-next-forward-progress. No corrective policy is permitted until
one interval is dominant.

The default-off `QWEN_DEC_LIFECYCLE_TRACE=1` instrumentation records event-only monotonic
timestamps for every bounded lane job: `READY`, `ENQUEUE`, `START`, `DONE`,
`COMPLETE_SEEN`, `RESUME`, and `NEXT_PROGRESS`. It also records slot, active-stream count,
pending frames, decoder target, cohort size and the preceding consecutive mailbox
pause-run. `QWEN_SD_SCHED` is off; bounded async (`QWEN_SD_LANE_NONBLOCK=2`), decoder
targets, cohorts, topology, kernels and admission remain unchanged.

The active run is a single C10 4x8 diagnostic with the usual 60-second warm-up and
60–120-second measured window. It is non-qualifying because tracing changes observability
and uses a dirty diagnostic binary. Its analysis must split the lifecycle samples by B=2
versus B=3, singleton versus paired decode, and short versus long pause-run. The decision
rule is fixed in advance: queue delay implicates lane dispatch; compute implicates decoder
work/cohort shape; done-to-seen implicates completion signaling; seen-to-resume or
resume-to-progress implicates the main scheduler. If none dominates but long pause-runs
track singleton decode, inspect cohort formation/state ownership next. Raw logs remain
private; this addendum records only the eventual aggregate result.

### Result — lifecycle attribution

The C10 bounded-async baseline completed 120 requests with zero errors, rejects or
timeouts. Its traced window remains diagnostic-only (`STREAM_RTF p95 1.058`, TTFA p95
312 ms, safe-start p95 1.049 s, stall@250 10.0%, stall@500 3.6%). Resource samples were
stable (thread count unchanged; RSS/FDs did not grow).

All values below are event-to-event milliseconds, p50/p95/max. `pause<=16` and
`pause>=64` are central-loop-turn buckets; they are descriptive rather than a production
threshold.

| split | n | ready→enqueue | enqueue→start | compute | done→seen | resume→progress |
|---|---:|---:|---:|---:|---:|---:|
| all | 6,061 | 0.01/0.03/328.90 | 48.38/136.53/270.14 | 80.20/266.75/319.53 | 15.44/33.77/314.54 | 35.17/39.68/301.54 |
| B=2 | 3,181 | 0.01/0.03/269.05 | 36.30/55.56/157.02 | 79.95/144.85/268.77 | 18.01/32.54/140.36 | 34.28/38.77/301.54 |
| B=3 | 2,738 | 0.01/0.03/328.90 | **109.90/213.77/270.14** | 80.55/269.47/319.53 | 12.84/35.17/314.54 | 35.79/39.89/46.63 |
| singleton | 5,357 | 0.01/0.03/328.90 | 48.32/140.78/270.14 | 79.88/86.33/129.45 | 18.73/34.26/314.54 | 35.22/39.65/301.54 |
| pair | 704 | 0.01/0.03/0.04 | 50.16/62.38/66.09 | **265.50/272.05/319.53** | 0.55/1.02/111.39 | 34.62/40.06/44.48 |
| pause<=16 | 5,131 | 0.01/0.03/313.45 | 47.16/131.13/267.95 | 79.94/88.53/268.11 | 17.39/33.61/199.20 | 35.10/39.46/191.79 |
| pause>=64 | 878 | 0.01/0.03/328.90 | 53.78/**223.41**/270.14 | **261.85/271.85/319.53** | 0.72/34.66/314.54 | 35.76/40.63/301.54 |

**Verdict: lane queue/dispatch plus cohort workload is CONFIRMED as the next causal
branch.** The B=3 incremental penalty is enqueue-to-lane-start: p95 rises by roughly
158 ms versus B=2. Completion visibility and post-resume forward progress are not the
primary tail (both are about 35–40 ms p95). Long pause runs also coincide with ~262–272 ms
pair compute, while singleton compute is about 86 ms p95. The result rejects a pure
completion-signal/polling explanation and does not justify another urgency-weight policy.

The next experiment must isolate lane queue/dispatch from paired decoder workload, with
one change only. Do not alter admission, kernels, topology or global scheduler priority.

## Singleton decoder discriminator

The next one-change diagnostic disabled paired decoder cohorts by applying
`QWEN_SD_MULTISLOT=1`. Bounded async remained `QWEN_SD_LANE_NONBLOCK=2`; decoder targets,
admission, 4x8 topology, kernels and scheduler were unchanged. The lifecycle trace was
kept enabled. This is a diagnostic comparison, not a qualification run.

| arm | completions | STREAM p95 | TTFA p95 | safe-start p95 | stall@250 | stall@500 |
|---|---:|---:|---:|---:|---:|---:|
| paired baseline | 120 | 1.058 | 312 ms | 1049 ms | 10.0% | 3.6% |
| singleton decoder | **126** | **0.829** | **159 ms** | **415 ms** | **0%** | **0%** |

Lifecycle attribution moved in the same direction:

| split | paired baseline p95 | singleton p95 |
|---|---:|---:|
| B=3 enqueue-to-lane-start | **213.77 ms** | **135.13 ms** |
| B=3 decoder compute | **269.47 ms** | **83.49 ms** |
| B=2 enqueue-to-lane-start | 55.56 ms | 65.02 ms |
| B=2 decoder compute | 144.85 ms | 86.22 ms |

The singleton trace contained 6,533 singleton groups and no pair groups. Completion
visibility and resume-to-progress remained approximately 35–40 ms p95, so the improvement
is not explained by changing the wake path. **CONFIRMED:** paired cohort execution is
causally poisoning the B=3 lane: it inflates both pair compute and the queue-to-start
tail, and removing it restores the C10 playback KPIs in this diagnostic.

This does not justify globally disabling cohorts as a production default: singleton
execution may sacrifice useful throughput at healthier shapes. The next design question
is dynamic cohort admission—permit a pair only when its measured cost/queue slack is safe,
otherwise dispatch singleton work—using one isolated A/B at a time. No kernel change is
indicated by this result.

## Paired-cohort cost: static path audit and isolated microbench

### Static audit — what changes between ng=1 and ng=2

`dec_worker_main` (`qwen_tts.c:2735`) branches on the group size into **two separately
written decoder implementations**, not one kernel parameterised by batch:

* `ng == 1` -> `qwen_speech_decoder_decode_streaming_st` -> `sd_stream_st_body`
  (`qwen_tts_speech_decoder.c:3122`).
* `ng > 1`  -> `qwen_speech_decoder_decode_streaming_batch` -> `sd_stream_batch_body`
  (`:4508`). `nb == 1` inside that body falls straight back to the per-item path
  (`:4530`), so the group size alone selects the implementation.

Cohort formation itself is clean and is **not** the cost. `dec_enqueue_cohort`
(`qwen_tts.c:2920`) pairs only slots with identical `nframes`, uses preallocated slot
jobs, and the cohort is always flushed inside the same frame turn (`qwen_tts.c:4180`):
no cross-turn holding, no extra queue turn, no second mailbox wake. Parallelism is also
unchanged -- `qwen_conv1d_int8_v2_ctx` and `qwen_conv1d_int8_v2_multi_ctx_strided`
(`qwen_tts_kernels.c:10666`, `:10789`) derive `n_blocks` from the **per-slot** length with
the same `tb` formula and run on the same `sd_pool_run` team.

Under this profile (`RES1_V2=1`, `SD_INT8=1`, AMX off, `CONVT_STACK`/`CNEXT_I8`/`GLUE`
unset) the two paths also select equivalent math: the same int8 SDOT leaf for pre-conv,
initial conv, res1 and res2 (`rag_conv1d`'s multislot branch at `:3985` reuses the *same*
`sd_wq` cache entry as the per-item path), the same f32 ConvT algorithm (`rag_convt`
`:4173` is the k-GEMM+scatter of `causal_conv_transpose1d_blas` `:1071`), the same f32
`convnext_mlp`, and `cblas_sgemm` transformers. `rag_conv1d_fused_residual` (`:4088`)
requires `sd_amx_d_enabled()` and therefore always returns 0 on Arm.

So the pair does structurally ~2x the work of a singleton. The measured 3.1x is not
decoder math.

**One clearly accidental difference was found.** `sd_stream_st_body` installs a persistent
per-slot bump arena (`SD_ARENA_BLOCK_MIN` 32 MB, comment: *"one 8-frame chunk of the conv
stack needs tens of MB"*, `:72`) and resets it per call. `sd_stream_batch_body` **never
sets `g_sd_arena`**, so `sd_tmp_alloc` degrades to `posix_memalign` and `sd_tmp_free` to
`free` (`:88-104`):

| path | arena sites | raw `posix_memalign` sites |
|---|---:|---:|
| per-item body + conv stack | 22 | 3 |
| ragged body + conv stack + `rag_*` helpers | 2 (both degrade to malloc) | 31 |

The 6 ragged conv-stack sites sit inside `4 upsample blocks x 3 res blocks`, so one pair
decode performs roughly 40 allocate/first-touch/free cycles on buffers that grow x1920
through the stack -- on the order of 115 MB of churn per call at a 4-frame quantum.

### Isolated microbench — the allocation hypothesis is REFUTED

`tests/decode_quantum_bench.c` calls the same entry point with no server around it.
Graviton4, `taskset -c 0-7`, 8 threads, the `aws-c8g-8xlarge-32c-arm-v2-all-on` profile
environment, same 0.6B model as the C10 diagnostic, p50 of 10 warm calls after 4 warm-ups.
Group 1 is the per-item path; groups 2/3 take the ragged path when multislot is on.

`chunk = 4` (the product quantum), call wall time in ms:

| group | A `MULTISLOT=0` (sequential singletons) | B `MULTISLOT=2` (ragged cohort) | C = B + tuned glibc allocator |
|---:|---:|---:|---:|
| 1 | 40.6 | 40.1 | 40.0 |
| 2 | **81.0** | **132.1** | 132.4 |
| 3 | 120.0 | 192.8 | 184.4 |

`chunk = 8`: group 1 `76.3 / 75.0 / 74.9`, group 2 `150.3 / 256.8 / 244.3`,
group 3 `225.7 / 378.1 / 355.9`. Arm C set `MALLOC_MMAP_THRESHOLD_=256M`,
`MALLOC_TRIM_THRESHOLD_=1G`, `MALLOC_TOP_PAD_=128M`, i.e. glibc approximating the arena
with no code change.

Readings:

1. Group 1 is identical across all three arms (40.6/40.1/40.0): a clean control proving
   only the ragged path changed.
2. Arm A is exactly linear (`81.0 ~= 2 x 40.6`). The cohort costs **1.63x two sequential
   singletons**, i.e. 3.25x one singleton -- reproducing the server's `265 / 86 = 3.08x`
   with no serving noise.
3. **Arm C recovers 0% at chunk 4 and about 5% at chunk 8.** The missing arena is a real
   code difference but is **not** the cost. **Allocation churn is REFUTED as the cause.**

**CONFIRMED: the paired cohort is intrinsically expensive on the Arm ragged path.**
A further finding the server could not give: the cohort is a **pure loss at B=2 as well,
at zero queue pressure** (132.1 vs 81.0 ms). There is no measured operating point on this
Arm path where cohort formation pays; the shared-weight benefit of the multi kernel is
more than cancelled by the ragged implementation.

Therefore option B (fix an accidental pair-path pathology) is closed and option A
(dynamic cohort admission) is the next experiment. Do not re-run the allocator arm, and
do not open a ragged-path kernel project on this evidence alone.

An earlier attempt at this microbench is **void and must not be cited**: the profile
environment is emitted as one comma-separated line, and `env $BASE ...` therefore set a
single malformed variable instead of 46. With `QWEN_SD_INT8`/`QWEN_SD_RES1_V2` absent,
`sd_multislot_slots()` returned 0 in every arm and all three ran the same f32 path
(group 1 chunk 4 = 344 ms, arms A and B identical to 0.1 ms).

### A — dynamic cohort admission (crude first policy)

Implemented as `QWEN_SD_COHORT_MAX_B`, default 0 = current unconditional cohort
behaviour. When set to N, a cohort may form only while the worker has at most N active
streams; above that, decoder units dispatch as singletons immediately. The decision is
taken once per frame turn from `n_active`, so a cohort is never left half-built across the
cap boundary. Nothing else changes: admission, kernels, topology, bounded async, decoder
targets and scheduler are untouched.

### C10 A/B result — cohort cap 2

Both arms on the same new binary, one variable (`QWEN_SD_COHORT_MAX_B`), same Arm-v2
all-on profile, 4x8, 0.6B diagnostic model, 60 s warm-up and one 60 s measured window.
Bounded async (`QWEN_SD_LANE_NONBLOCK=2`), lifecycle trace and stage trace were on in
both. Diagnostic comparison, not a qualification: one window per arm, so the runner
reports `LATENCY KPI NOT_ASSESSED` and `SOAK RESULT PARTIAL` in both.

| metric | control, cap off | treatment, cap 2 |
|---|---:|---:|
| STREAM p50/p95 | 0.579 / **1.035** | 0.669 / **0.847** |
| TTFA p50/p95 | 112.1 / 301.7 ms | 113.4 / 307.8 ms |
| safe-start p50/p95 | 186 / **1032 ms** | 200 / **504 ms** |
| stall@250 / @500 | 11% / 2% | **0% / 0%** |
| max_gap p95 | 0.471 s | 0.458 s |
| completed | 133 (130.7 s) | 122 (126.2 s) |
| errors / rejects / timeouts | 0 / 0 / 0 | 0 / 0 / 0 |
| resource stability | PASS | PASS |

Lifecycle attribution confirms the mechanism fired exactly as designed and nothing else
moved:

| split | control | treatment |
|---|---:|---:|
| B=3 cohort groups | **460** | **0** |
| B=2 cohort groups | 32 | 108 (preserved) |
| B=1 cohort groups | 0 | 0 |
| B=3 enqueue->start p50/p95 | 117.36 / **213.59 ms** | 115.72 / **134.93 ms** |
| B=2 enqueue->start p95 | 54.89 ms | 59.69 ms |

The treatment's B=3 enqueue->start p95 of 134.93 ms reproduces the 135.13 ms of the
earlier singleton-only (`MULTISLOT=1`) arm, so the cap recovers the full B=3 benefit while
keeping B=2 cohorts alive.

**KEEP, as a default-off flag.** Playback continuity is restored at C10 (STREAM p95 under
one, both stall buckets at zero, safe-start halved) at flat TTFA, zero functional failures
and stable resources.

Costs and open items, not to be lost:

* Completions fell 133 -> 122; normalised for the slightly shorter treatment window that
  is about -5% throughput. The cap buys continuity with a real throughput cost.
* STREAM p50 rose 0.579 -> 0.669 while p95 fell. Consistent with trading a little per-unit
  sharing for removal of the tail.
* **Unexplained, recorded as an observation only:** in the control, *singleton* compute at
  B=2 shows a 267.42 ms p95 tail that disappears in the treatment (87.85 ms), while B=3
  singletons look clean in both (91.14 / 89.41 ms). Candidate explanations are
  cross-worker cache/fabric interference from cohort decodes on other workers -- which
  would be consistent with ARM-SOAK-4a/4d -- or a slot-keyed attribution artifact in the
  trace parser. Do not cite it as a finding until one of the two is tested.
* One window per arm. Before any promotion this needs the multi-window soak and the
  per-class KPI, not another crude-policy tweak.

### C8 control check — the cap is inert where it should be

Same binary, same one variable, same runner at `--concurrency 8`.

| metric | control, cap off | treatment, cap 2 |
|---|---:|---:|
| STREAM p50/p95 | 0.529 / 0.619 | 0.526 / 0.627 |
| TTFA p50/p95 | 112.7 / 129.2 ms | 112.2 / 140.1 ms |
| safe-start p50/p95 | 137 / 202 ms | 147 / 214 ms |
| stall@250 / @500 | 0% / 0% | 0% / 0% |
| max_gap p95 | 0.250 s | 0.252 s |
| completed | 135 (126.0 s) | 137 (125.2 s) |
| errors / rejects / timeouts | 0 / 0 / 0 | 0 / 0 / 0 |
| resource stability | PASS | PASS |

**NO REGRESSION.** Every delta is inside single-window noise (STREAM p95 +0.008,
safe-start +12 ms, TTFA p95 +11 ms, completions +2).

The lifecycle trace explains why the result is null rather than merely small: at C8 on
4x8 the per-worker occupancy is essentially B=2 (6,030 of 6,409 singleton starts at B=2;
only 51 events at B=3), so the cap has almost nothing to act on -- 8 B=3 cohorts in the
control become 0 in the treatment. **The mechanism is correctly inert at the already-good
operating point.**

### Server-side confirmation of the microbench

The C8 traces confirm the isolated microbench on the live server, independently of the
C10 knee. B=2 cohorts are preserved by design in both arms, and their compute p95 is
`251.19 ms` (control) and `263.87 ms` (treatment) against a B=2 *singleton* compute p95 of
`85.86 / 88.95 ms`. The same ~3x per-cohort penalty the bench measured at zero queue
pressure is therefore present in production shape at B=2 as well.

### Verdict and the open decision

`QWEN_SD_COHORT_MAX_B=2` is **KEEP as a default-off diagnostic**: it removes the C10 B=3
playback knee (STREAM p95 `1.035 -> 0.847`, safe-start p95 `1032 -> 504 ms`,
stall@250/@500 `11%/2% -> 0%/0%`) at flat TTFA and is inert at C8. It is not promoted:
one measured window per arm, `SOAK RESULT PARTIAL`, no per-class KPI, no audio gate.

**The open question this raises, and deliberately does not answer:** both the microbench
and the C8 server trace say a cohort is a net loss at B=2 too, so the natural next
one-change A/B is `COHORT_MAX_B=1` (never cohort) against `COHORT_MAX_B=2`. Note the
earlier `MULTISLOT=1` singleton-only arm reached STREAM p95 `0.829`, TTFA p95 `159 ms`,
safe-start p95 `415 ms` and 126 completions at C10, i.e. better than cap 2 on every one of
those, though against a different control run. If cap 1 wins, the honest conclusion is
that the shared-weight cohort mechanism has no operating point on the Arm ragged path and
should be retired there rather than tuned -- which is a product decision, not a tuning
step, and must not be taken from a single 60 s window.

Do not jump from here to a queue-slack or playback-slack formula: the crude occupancy cap
has not yet been compared against its own simplest competitor.

## Architectural review — retire the Arm ragged cohort path, or keep it at B=2?

Scope of this review: one question, adversarial, no code change. Every measurement below
ran on the AWS Graviton4 box (`taskset -c 0-7`, 8 threads, the
`aws-c8g-8xlarge-32c-arm-v2-all-on` profile environment, the same 0.6B diagnostic model as
the C10 runs, `tests/decode_quantum_bench.c` patched on the box only to select one cell,
12 warm calls after 4 warm-ups). Raw evidence is private under
`arm-evidence/g4-decoder-cohort-{costmap,phase,smallchunk}-20260914*`. Source was read on
the identical tree.

### 1. Where the penalty is — three independent decompositions agree

**Cost map** (`QWEN_COST_MAP=1`, inclusive regions, ms per bench call, chunk 4):

| region | 1 singleton | 2 sequential singletons | ragged pair | pair / sequential |
|---|---:|---:|---:|---:|
| decoder.total | 44.55 | 86.67 | **136.16** | 1.57 |
| decoder.conv_stack | 42.85 | 83.40 | **133.26** | **1.60** |
| decoder.transformer | 1.11 | 2.24 | 1.87 | 0.83 |
| pre_conv + vq + in/out_proj | 0.57 | 1.00 | 1.01 | 1.01 |

98 % of the pair's cost and 100 % of its excess (+49.9 ms) sit in `conv_stack`. The parts
the ragged path *shares* across slots (transformer GEMMs) are at parity or slightly better.

**Phase split** (`QWEN_SD_PHASE=1`, `[SDUP]` per call; the ragged body never resets the
`sd_up_*` accumulators, so its values were differenced call-to-call):

| phase | 2 sequential singletons | ragged pair | delta |
|---|---:|---:|---:|
| **res1** (int8 dilated k=7 conv, 12 calls) | 29.67 | **72.15** | **+42.5** |
| **res2** (int8 1x1 conv, 12 calls) | 6.82 | **13.93** | **+7.1** |
| convt (f32 k-GEMM + scatter) | 18.80 | 18.09 | -0.7 |
| resadd | 3.08 | 3.06 | 0.0 |
| snake | 6.27 | 2.93 | (asymmetric instrumentation; small) |

`res1 + res2` account for +49.6 of the +49.9 ms conv-stack excess. Those two phases are
precisely the calls that dispatch `qwen_conv1d_int8_v2_multi_ctx_strided` instead of the
single-slot `qwen_conv1d_int8_v2`. The f32 ConvT, which the ragged path runs as one GEMM
over both slots, is 0.7 ms *cheaper* -- the weight-sharing premise does work where it is
realised; it is simply worth very little. Everything the ragged glue does (allocation,
`rag_dwconv` copies, transposes, ragged offsets) is bounded at roughly 10 ms by the same
table and by the f32 control (below).

**Disassembly of the two SDOT workers** (`objdump -d qwen_tts_kernels.o` on the box):

| worker | `sdot` | q-register stores to `[sp]` | q-register loads from `[sp]` |
|---|---:|---:|---:|
| `sd_dconv_worker` (single slot, named `a00..a33`) | 16 | 4 (prologue) | **0** |
| `sd_dconv_multi_worker` (`acc[3][4][2]`, runtime `S`/`mn` bounds) | 30 | **40** | **33** |

The multi kernel's accumulator arrays are indexed by runtime loop variables, so GCC keeps
them on the stack: about one stack round-trip per `sdot`. The single kernel's accumulators
never leave registers. The DL-4 design (`.work/dl4-multislot-design-20260910.md` §5) named
register pressure as the risk and chose the 4x2 tile to stay under `8*S+4` vectors; the
compiler did not keep even that tile in registers.

**f32 control, recovered from the void first microbench.** That run had `QWEN_SD_INT8`
absent, so groups >= 2 took the *same ragged glue* with f32 im2col+SGEMM convs. There the
pair was `695.6 ms` vs `687.4` for two singletons at chunk 4 (1.01x), `1381 / 1272` at
chunk 8 (1.09x) and it *won* at chunk 1 (`197 / 258`). The ragged glue is therefore not
the cost; what differs between f32-ragged and int8-ragged is the multi kernel.

### 2. The cohort's theoretical ceiling, from shapes

The only benefit the mechanism was designed for is reading each weight once per cohort.
Per decoder unit on this path: int8 conv weights ~31 MB (res1 16.5, initial 11, res2
2.3, pre 1.5), f32 ConvT + ConvNeXt pointwise ~157 MB. At Graviton4's measured ~108 GB/s
per worker at 4x8 that is ~1.7 ms per unit; at Graviton5's ~36 GB/s, ~5 ms. The ragged
path already banks the f32 share (`convt -0.7 ms` above). **The cohort's ceiling is a few
milliseconds per pair against a measured 50 ms loss**, on either Arm host.

### 3. Every regime, measured on the box

| cell | singleton | 2 sequential | ragged pair | pair / sequential |
|---|---:|---:|---:|---:|
| chunk 1 | 18.1 | 36.2 | 57.4 | **1.59** |
| chunk 2 | 25.7 | 51.4 | 84.6 | **1.65** |
| chunk 4 | 40.6 | 81.0 | 132.1 | **1.63** |
| chunk 8 | 76.3 | 150.3 | 256.8 | **1.71** |
| group 3, chunk 4 | -- | 120.0 | 192.8 | **1.61** |
| group 3, chunk 8 | -- | 225.7 | 378.1 | **1.67** |

The penalty is a near-constant ~1.6x of the sequential cost across chunk 1-8 and S=2-3:
per-element, not per-call. The fixed-cost regime (1-2 frames), the one place a batched
call could plausibly win, loses by the same factor.

### 4. Attempts to falsify "retire", and what each ran into

1. *Bandwidth-starved host (Graviton5) flips it.* Ceiling arithmetic: <= ~5 ms/pair of
   sharing versus a 50 ms loss that is compute/register-bound and does not shrink with
   bandwidth. No.
2. *Small units win on fixed cost.* Measured: chunk 1 and 2 lose 1.59x / 1.65x. No.
3. *Fewer lane turns / mailbox wakes / elastic-width transitions.* The microbench has
   none of these and the pair still loses by ~50 ms; a saved turn would have to cost
   > 25 ms. The C10 lifecycle trace also showed done->seen and resume->progress at ~35-40
   ms p95 in *both* the paired and the singleton arms: those are per-unit main-loop
   latencies that pairing does not reduce. No.
4. *Three-slot cohorts amortise better.* 1.61x / 1.67x. No.
5. *The kernel fix would make it win.* Best case for a spill-free S=2 kernel is res1 back
   to ~30 ms, i.e. pair ~85-90 ms vs 81 sequential -- parity plus the few-ms ceiling. The
   fix can remove the loss; it cannot create a win. No.
6. *Turin proved cohorts.* The DL-4 §4 unit-cost gate (`conv_up` at 1 slot vs 2 slots)
   is not recorded as passed anywhere in `.work`; the Turin product profile's "verified
   2-slot cohort" rests on a combined lane+V2+cohort smoke, never an isolated arm. Not
   evidence for Arm, and an open question for x86 (below). No.
7. *First-chunk / TTFA priority.* A cohort leader inherits `first` and goes to the queue
   head, but so does every singleton `first` unit. No difference. No.
8. *1.7B changes the picture.* The speech decoder is identical across 0.6B/1.7B. No.

Nothing survived.

### 5. Answers

**Is there any current server-side condition where the existing ng>1 ragged path can
plausibly outperform sequential singleton decode?** No. Every measured regime loses by
~1.6x, the mechanism's theoretical upside is bounded at a few ms per pair by weight
sizes and measured bandwidth, and the scheduling-side savings would need to exceed 25 ms
per pair to matter.

**Is the penalty fundamental to the implementation structure, or is there one specific
fix worth testing before retiring?** It is not fundamental to the ragged structure -- the
glue is at parity (f32 control 1.01x, transformer/convt at or below sequential). It is one
concrete accidental pathology: `sd_dconv_multi_worker`'s runtime-indexed accumulator
arrays spill to the stack (40 stores / 33 loads around 30 `sdot`, versus 0 loads in the
single kernel), and that kernel carries 49.6 of the 49.9 ms excess. The specific fix is an
`S == 2` specialisation with named accumulators (16 `int32x4` + 4 activation + 1 weight
vector = 21 registers, the single kernel's budget), same k-loop and same reduction order,
gated by the existing `--self-test` multi cases (S=2 and S=3 pass against the single-slot
oracle today) and by `QWEN_SD_PHASE` res1 <= 2x single. **It is worth testing only if a
reason to keep cohorts exists, because its best outcome is parity.** It is not a reason to
keep them. The x86 VNNI twin has the identical array structure (`acc[3][4][2]`,
`facc[3][4][2]`, `xv[3][2]` over runtime `S`); it was not disassembled here and must not
be touched by this track, but the Turin cohort claim deserves the same three-cell
microbench when an x86 box is next rented.

**Would `COHORT_MAX_B=1` be the cleanest production policy on Arm for now?**
Behaviourally it is "no cohorts" (a cohort needs two active streams), and it is the right
*decision arm* because it is a one-variable change on the same binary. The cleanest
*production expression* is `QWEN_SD_MULTISLOT=0` in the Arm profile: it is the pre-existing,
preflight-valid fallback (`multislot: "VALID FALLBACK"` in `arm-product`), it removes the
whole int8 ragged dispatch rather than arming a dead mechanism, it adds no new code to the
product path, it leaves the Turin profile (`MULTISLOT=2`, no cap) untouched, and it keeps
`QWEN_SD_COHORT_MAX_B` purely diagnostic.

**What exact one-variable A/B makes that decision defensible?** Same binary, same private
G4 runner, both arms with `QWEN_SD_LANE_NONBLOCK=2,QWEN_DEC_LIFECYCLE_TRACE=1,
QWEN_STAGE_TRACE=1`; arm A `QWEN_SD_COHORT_MAX_B=2`, arm B `QWEN_SD_COHORT_MAX_B=1`; run
at C8 first (the B=2 regime, where the cap actually bites) and then C10; 2 minutes each.
Mechanism proof: the lifecycle trace must show B=2 cohorts > 0 in A and 0 in B. Decision
rule, fixed in advance: keep B=2 cohorts only if A beats B on STREAM p95 **and** stall@250/
@500 **and** completions at **both** points by more than the observed single-window noise
(from the C8 control/treatment pair, where the cap was inert: ~0.01 STREAM p95, ~12 ms
safe-start, ~11 ms TTFA, +-2 completions). Expected effect size at C8 is ~1.5-2 % of lane
time (82 cohorts x ~50 ms over 4 workers x 60 s), so a null result is the *expected*
outcome and counts against the cohort, not for it: the burden of proof is on a mechanism
with a measured per-call loss and a few-ms ceiling. If A does not win above noise, retire
on Arm by setting `QWEN_SD_MULTISLOT=0` in the Arm profile; the flag, kernel and self-test
cases stay in the tree, default-off, for x86 and for a future exact kernel fix.

### The decision A/B — `COHORT_MAX_B=2` vs `=1` at C8 and C10

Run exactly as pre-registered: same binary, private G4 runner, both arms with
`QWEN_SD_LANE_NONBLOCK=2,QWEN_DEC_LIFECYCLE_TRACE=1,QWEN_STAGE_TRACE=1`, C8 first then C10,
60 s warm-up + one 60 s window each, four runs back to back (2026-09-14 19:34-19:43 UTC).
Diagnostic, not qualification (`SOAK RESULT PARTIAL` in all four).

| point | arm | STREAM p50/p95 | TTFA p95 | safe-start p50/p95 | stall@250/@500 | max_gap p95 | completed | err/rej/to |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C8 | A cap 2 | 0.530 / 0.625 | 134.7 ms | 167 / 206 ms | 0% / 0% | 0.251 s | 137 | 0/0/0 |
| C8 | B cap 1 | 0.528 / **0.612** | 156.8 ms | 171 / 227 ms | 0% / 0% | 0.245 s | 133 | 0/0/0 |
| C10 | A cap 2 | 0.591 / 0.831 | 196.8 ms | 192 / 476 ms | 0% / 0% | 0.408 s | 124 | 0/0/0 |
| C10 | B cap 1 | 0.596 / 0.835 | **160.2 ms** | 219 / **431 ms** | 0% / 0% | 0.402 s | **129** | 0/0/0 |

Resource stability PASS in all four; zero queue rejections and server timeouts.

Mechanism proof from the lifecycle traces:

| point | arm | B=2 cohorts | B=3 cohorts | B=2 pair compute p50/p95 | B=2 singleton compute p50/p95 |
|---|---|---:|---:|---:|---:|
| C8 | cap 2 | **82** | 0 | 261.5 / 270.4 ms | 80.7 / 89.1 ms |
| C8 | cap 1 | **0** | 0 | -- | 80.5 / 87.7 ms |
| C10 | cap 2 | **56** | 0 | 80.1 / 260.2 ms | 80.5 / 86.9 ms |
| C10 | cap 1 | **0** | 0 | -- | 80.5 / 85.4 ms |

The arms differ only in whether B=2 cohorts exist, and the ~3x per-cohort penalty is again
visible in production shape. With 56-82 cohorts per window the removable lane time is
about 1.5 % (C8: 82 x ~100 ms p50 excess over 4 workers x 125 s), which is the predicted
effect size and explains why the KPI deltas are small and mixed.

**Applying the pre-registered rule** -- keep B=2 cohorts only if cap 2 beats cap 1 on
STREAM p95 *and* stall@250/@500 *and* completions at *both* points, by more than
single-window noise (~0.01 STREAM p95, ~12 ms safe-start, ~11 ms TTFA, +-2 completions):

* STREAM p95: cap 1 better at C8 (0.612 vs 0.625), tie at C10 (0.835 vs 0.831). Not met.
* stall@250/@500: 0/0 in all four. Tie.
* completions: cap 2 +4 at C8, cap 1 +5 at C10. Not met at both points.

**RULE NOT MET -> RETIRE cohorts on Arm.** Stated honestly, the picture is mixed rather
than null: at C8 cap 2 is ahead on TTFA p95 (-22 ms), safe-start p95 (-21 ms) and
completions (+4), each just above the noise floor; at C10 the sign reverses (cap 1: TTFA
p95 -37 ms, safe-start p95 -45 ms, +5 completions). A signal that flips sign between two
adjacent operating points at ~2x the noise floor is noise, and it matches the ~1.5 %
effect size expected from the cohort counts. Nothing in four windows shows a consistent
win for the mechanism that the microbench, the cost map, the phase split and the
disassembly all show to be a per-call loss with a few-ms ceiling.

**Recommended production expression (not applied here -- it is a qualification-level
change):** set `QWEN_SD_MULTISLOT=0` in `configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json`
(and any other Arm profile carrying `=2`), i.e. the pre-existing preflight-valid fallback;
keep `QWEN_SD_COHORT_MAX_B` as a default-off diagnostic; leave the Turin profile and the
VNNI kernel untouched. Promotion still requires the multi-window soak and the per-class KPI
on that profile; this A/B is the reason to run it, not a substitute for it.

## PRODUCTION DECISION — cohorts retired in the Graviton4 profile, and the qualification that followed

### The change

`configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json`: `QWEN_SD_MULTISLOT` `"2" -> "0"`, with the
measurement record in `why` and a `revert` condition. The `parity.notes` sentence claiming the
cohort was active was corrected. Nothing else changed: `arm-product`, `axion-*`,
`turin-c8a-32c-vnni-product` (still `=2`) and `turin-c8a-32c-vnni-control` are untouched, and
`QWEN_SD_COHORT_MAX_B` remains a default-off diagnostic that appears in no profile. The
profile's parity contract already listed `multislot: ["ACTIVE", "VALID FALLBACK"]`, so the
strict preflight stays valid with the cohort off.

### Qualification, 2026-09-15, AWS Graviton4 c8g.8xlarge

One campaign, `tmux`, profile used exactly as committed; evidence under
`arm-evidence/g4-multislot0-qualification-20260915T073728Z`. Model, bank, speaker and
4x8/cap-8 layout identical to the diagnostics, so C8 is a true regression control.

Gates: `--caps`, `--self-test` (0 failures), `--dispatch-map`, strict profile preflight — all
PASS. Preflight recorded `profile_valid: true`, `multislot_active: false`,
`feature_status.multislot: "VALID FALLBACK"`, `resolved_decoder_mode: "per-item-int8-dotprod"`.

| 30-min closed-loop soak | C8 (regression control) | C10 (recovered point) |
|---|---:|---:|
| SOAK RESULT | **PASS** | **PASS** |
| completed / errors / rejects / timeouts | 3805 / 0 / 0 / 0 | 3892 / 0 / 0 / 0 |
| windows | 5 x 300 s | 5 x 300 s |
| STREAM RTF p50/p95 | 0.510 / **0.562** | 0.68 / **0.805** |
| TTFA p50/p95 | 112.2 / 151.3 ms | 111.7 / 190.2 ms |
| safe-play-start p50/p95 | 130 / **179 ms** | 189 / **361 ms** |
| max_gap p95 | 0.246 s | 0.320 s |
| stall@100 / @250 / @500 / @1000 | 0.08% / **0%** / **0%** / **0%** | 11% / **0%** / **0%** / **0%** |
| LATENCY KPI | PASS | PASS |
| PER-CLASS KPI (5 classes) | all PASS | all PASS |
| RESOURCE STABILITY | PASS (threads flat 92, RSS +0.05%/116 samples) | PASS |

### Paired against the previous generation — same profile, same commit, one variable

The previous private measurement generation ran on this host at commit `97c0fa1` with
`multislot_active: true` / `QWEN_SD_MULTISLOT=2` (verified in that run's own
`profile-preflight.json`, retained privately), same model, bank and layout. The new binary is `97c0fa1-dirty` — the diagnostic additions
(`QWEN_DEC_LIFECYCLE_TRACE`, `QWEN_SD_COHORT_MAX_B`) are default-off and inactive here.

0.6B, 30-minute closed-loop:

| metric | C8 cohort ON (v1) | C8 cohort OFF | C10 cohort ON (v1) | C10 cohort OFF |
|---|---:|---:|---:|---:|
| status | PASS | **PASS** | **OVER LIMIT** | **PASS** |
| STREAM RTF p50/p95 | 0.514 / 0.801 | 0.510 / **0.562** | 0.75 / **1.04** | 0.68 / **0.805** |
| TTFA p95 | 125.7 ms | 151.3 ms | 286 ms | **190.2 ms** |
| safe-start p95 | 216 ms | **179 ms** | 723 ms | **361 ms** |
| max_gap p95 | 0.347 s | **0.246 s** | -- | 0.320 s |
| stall@250 / @500 | 0.485% / 0% | **0% / 0%** | 19.3% / 2.2% | **0% / 0%** |
| completed | 3342 | **3805** (+13.9%) | 3247 | **3892** (+19.9%) |

**C10 moved from OVER LIMIT to a full 30-minute PASS.** The 0.6B sustained recommendation on
this host therefore moves C8 -> C10, a 25% density increase, with every playback gate met and
zero functional failures. At C8 the one metric that got worse is TTFA p95 (+25.6 ms); at C10
TTFA *improved* by 96 ms. Both are reported as measured.

No regression is attributable to singleton decode: C8, the control, improved on STREAM p95,
safe-start, max_gap, both stall buckets and throughput, and passed every KPI. Cohort tuning
therefore stays closed.

### Campaign exit status

The campaign script exited `1` solely because the first audio-gate step called
`tests/compare_audio.py` without `librosa` installed, so it returned non-zero on all four pairs
without comparing anything. That is a tooling gap, not an audio result: the four conc-1 WAV
pairs (cohort ON vs OFF) are **byte-identical** (md5 and size), which is stronger than the
`mel-corr >= 0.98` the gate asks for, and the multi-slot self-test cases (S=2 and S=3) pass
against the single-slot oracle on this box, so the retired kernel was numerically exact by
construction. Both soaks reported `SOAK RESULT: PASS` independently of that exit code. A
completed audio gate — librosa installed, plus a concurrency-4 paired comparison with an
ON-vs-ON noise floor so cohorts actually form in the control arm — was run afterwards; result
recorded below.

## Knee screens and the second qualification round (2026-09-15)

### New tool: `tests/soak_fast.py` / `make soak-fast`

Walking a capacity ladder with 30-minute soaks costs most of a day and spends most of it
re-proving that bad points are bad. `tests/soak_fast.py` drives the *same* `serve_soak.py`,
profile and KPI definitions at 30 s warm-up + 2x90 s measured per point, classifies each
point against a rule fixed in advance, stops at the first knee and names the point worth
qualifying. Ladder syntax `8,10,12` / `8:12` / `8:16:2`.

    CLEAR    stream_p95 <= 0.90 and stall@250 == 0 and stall@500 == 0 and 0 errors
    HEALTHY  stream_p95 <  1.00 and stall@250 <= 1% and stall@500 <= 0% and 0 errors
    KNEE     anything else (a functional failure is a KNEE whatever the RTF says)

It writes `screen_summary.json` carrying `"kind": "screen"` and `"is_qualification": false`,
and both the file header and the printed output repeat that a screen is never a gate: short
windows cannot assess drift, per-class tails or resource growth, so `serve_soak.py` reports
`PARTIAL` by construction. Registered in `docs/BENCHMARKING.md` as a canonical tool and as
step **H2** of the box qualification order, between WAVE screening and the canonical SOAK.

**Screen-versus-soak calibration — OPEN, and weaker than first written.** The first note here
claimed a screen reads about one concurrency step optimistic on Graviton4. That was an
*interpolation*, not a measurement: G4 has 30-minute points at 0.6B C8 (`0.562`) and C10
(`0.805`) and screens at C11 (`0.803`) and C12 (`0.829`), with no same-point pair. The one
DIRECT same-point comparison available contradicts it — on Axion, 1.7B C12 with cohorts ON
measured `stream_p95 0.67` in the previous 30-minute generation and `0.653` in this
campaign's 180 s screen, i.e. essentially equal. Treat screens as approximately faithful at
the same point until proven otherwise, and still choose the qualification point from the
highest CLEAR rather than the last non-knee point, because that costs nothing.

**RESOLVED 2026-09-15.** The Graviton4 qualifications produced the missing same-point pairs
and screens are faithful, not optimistic:

| same point | 180 s screen | 30-min soak | delta |
|---|---:|---:|---:|
| G4 0.6B C12 `stream_p95` | 0.829 | **0.8364** | +0.007 |
| G4 1.7B C10 `stream_p95` | 0.830 | **0.8307** | +0.001 |
| Axion 1.7B C12 (cohorts ON) | 0.653 | 0.67 (previous generation) | +0.017 |

Three same-point pairs on two hosts agree within 0.02. The "one concurrency step optimistic"
reading is **withdrawn**: it came from comparing *different* points (30-min C10 against
screens at C11/C12) and the apparent offset was the capacity curve, not a duration effect.
A `soak-fast` screen can therefore be trusted to pick the qualification point directly.

### Graviton4 knee screens, production profile (`MULTISLOT=0`)

| model | C | STREAM p50/p95 | TTFA p95 | safe p95 | stall@250 | verdict |
|---|---:|---:|---:|---:|---:|---|
| 0.6B | 11 | 0.734 / 0.803 | 191 ms | 356 ms | 0% | CLEAR |
| 0.6B | 12 | 0.736 / **0.829** | 195 ms | 352 ms | 0% | **CLEAR** |
| 0.6B | 13 | -- / **1.013** | -- | -- | -- | **KNEE** |
| 1.7B | 8 | 0.645 / 0.698 | 175 ms | 250 ms | 0% | CLEAR |
| 1.7B | 10 | 0.757 / **0.830** | 208 ms | 363 ms | 0% | **CLEAR** |
| 1.7B | 11 | 0.772 / 0.829 | 211 ms | 381 ms | 0.21% | HEALTHY |
| 1.7B | 12 | 0.778 / 0.842 | 209 ms | 382 ms | 0.20% | HEALTHY |

Knee: 0.6B at C13; 1.7B leaves CLEAR after C10 and is still HEALTHY at C12. Chosen for the
canonical 30-minute qualification: **0.6B C12** and **1.7B C10**, i.e. the highest CLEAR of
each, given the screen-vs-soak calibration above. 1.7B C11/C12 are left as HEALTHY screen
evidence, not promoted.

### Axion (GCP c4a, Neoverse-V2, 32 core) — cohort discriminator, ARM-SOAK-10 step 1

Same three-cell `decode_quantum_bench` used on Graviton4, profile env applied correctly,
`taskset -c 0-7`, chunk 4, p50 of 12 warm calls:

| cell | Axion | Graviton4 |
|---|---:|---:|
| 1 singleton | 34.6 ms | 40.6 ms |
| 2 sequential singletons | 70.1 ms | 81.0 ms |
| 1 ragged cohort | **84.3 ms** | **132.1 ms** |
| penalty vs sequential | **1.20x** | **1.63x** |

**CONFIRMED on a second Arm host: the ragged cohort is a per-call loss there too**, and the
two sequential singletons are again exactly linear (2 x 34.6 = 69.2 against 70.1 measured).
The magnitude is much smaller than on Graviton4, which explains why the Axion campaign
looked healthy with the cohort enabled: it was paying ~20% per pair, not ~63%. The committed
Axion profile is NOT changed on the strength of a microbench alone; a serving A/B (cohort-ON
control screen at the previously recommended point, then a cohort-OFF ladder) is running,
and the profile is only touched if that A/B agrees.


## Graviton4 v2 qualification — both operating points PASS (2026-09-15)

Canonical 30-minute closed-loop soaks, `--strict-kpi`, profile exactly as committed
(`QWEN_SD_MULTISLOT=0`), 5 x 300 s windows, 4x8 / cap 8, same model, bank and speaker as the
whole campaign. Points chosen from the knee screens (highest CLEAR of each size).

| | 0.6B C12 | 1.7B C10 |
|---|---:|---:|
| SOAK RESULT | **PASS** | **PASS** |
| completed / errors / rejects / timeouts | 3962 / 0 / 0 / 0 | 3665 / 0 / 0 / 0 |
| STREAM RTF p50/p95 | 0.740 / **0.836** | 0.762 / **0.831** |
| TTFA p50/p95 | 114.4 / 197.8 ms | 126.1 / 207.0 ms |
| safe-play-start p50/p95 | 282 / 356 ms | 251 / 364 ms |
| max_gap p95 | 0.324 s | 0.366 s |
| stall@100 / @250 / @500 | 22.7% / **0%** / **0%** | 23.4% / **0%** / **0%** |
| LATENCY KPI / PER-CLASS (5) / RESOURCE | PASS / PASS / PASS | PASS / PASS / PASS |
| threads first->last, RSS growth | 92 -> 92, +0.02% | 92 -> 92, -0.01% |

**Operating points on this host move from C8/C8 to 0.6B C12 (+50%) and 1.7B C10 (+25%).**

One honest qualifier for the report: `stall@100` is 22-23% at these densities against 3.06%
at the old 0.6B C8 point. The 250 ms and 500 ms fixed buffers are clean (0%), so these points
are safe for a client that prebuffers >= 250 ms, which is what safe-play-start p95 of
356/364 ms already implies. They are NOT a claim about a 100 ms buffer. State the buffer
assumption next to the operating point rather than quoting the stall table alone.

### Profiles updated after qualification (2026-09-15)

| profile | `QWEN_SD_MULTISLOT` | `preferred_concurrency` | basis |
|---|---|---|---|
| `aws-c8g-8xlarge-32c-arm-v2-all-on` | 2 -> **0** | `unspecified` -> **0.6B C12; 1.7B C10** | two 30-minute soaks, both PASS |
| `axion-c4a-highcpu32-0p6b-all-on` | 2 -> **0** | 16 (unchanged pending its soaks) | microbench 1.20x + two paired serving screens |
| `arm-product`, `axion-16c-ttfa` | already 0 | unchanged | — |
| `turin-c8a-32c-vnni-product` | **2, untouched** | unchanged | x86; needs its own isolated microbench first |

Every Arm profile now ships the per-item decoder. The G4 file's stale claim that it "does not
promote a concurrency point before the host-specific audio and soak gates pass" was replaced,
since those gates have now passed, and the `>= 250 ms` client prebuffer assumption travels with
the promoted points in `objective.preferred_concurrency_evidence`.

## Axion (GCP c4a-highcpu-32) v2 qualification — both points PASS (2026-09-15)

Canonical 30-minute closed-loop soaks, `--strict-kpi`, the committed Axion profile with the
cohort retired, 5 x 300 s windows, 4x8 / cap 8, same private model, bank and speaker as the
previous generation so the comparison is row-for-row.

| | 1.7B C16 | 0.6B C16 |
|---|---:|---:|
| SOAK RESULT | **PASS** | **PASS** |
| completed / errors / rejects / timeouts | 5207 / 0 / 0 / 0 | 5281 / 0 / 0 / 0 |
| STREAM RTF p50/p95 | 0.738 / **0.789** | 0.730 / **0.845** |
| TTFA p50/p95 | 137.3 / 175.3 ms | 88.2 / 194.8 ms |
| safe-play-start p50/p95 | 251 / 332 ms | 263 / 371 ms |
| max_gap p95 | 0.327 s | 0.317 s |
| stall@100 / @250 / @500 | 15.1% / **0%** / **0%** | 12.1% / **0%** / **0%** |
| LATENCY KPI / PER-CLASS (5) / RESOURCE | PASS / PASS / PASS | PASS / PASS / PASS |
| threads first->last, RSS growth | 216 -> 216, +0.03% | 216 -> 216, +0.04% |

**1.7B moves C12 -> C16 (+33%). 0.6B stays at C16 but gains a sustained gate it never had**:
the previous generation recommended C16 on true-wave evidence alone and had only soaked C20,
which it classified as an edge probe.

Two distinct causes, and the split matters for how much credit the cohort retirement gets:
the paired serving screens at the previous operating points show the decoder change alone is
worth about -17.5% safe-start and +4.6% completions at 1.7B C12 and -13.7% / +8.3% at
0.6B C16, while most of the headroom at 1.7B came from the ladder never having been walked
past C12. Do not attribute the whole C12 -> C16 move to the flag.

Axion audio gate: paired concurrency-4 comparison, cohort-ON control against the committed
cohort-OFF profile, plus an OFF-vs-OFF noise floor. 12/12 pairs PASS at `mel_corr 1.00000`
on both comparisons, 12 files clean on `wav_qc`. Same result as Graviton4: the retirement is
numerically free.

### Cross-host summary of the campaign

| host | size | previous point | qualified v2 | knee (screen) | cohort penalty (microbench) |
|---|---|---|---|---|---|
| Graviton4 c8g.8xlarge | 0.6B | C8 | **C12** | C13 | 1.63x |
| Graviton4 c8g.8xlarge | 1.7B | C8 | **C10** | C11 (stalls appear) | 1.63x |
| GCP c4a-highcpu-32 | 1.7B | C12 | **C16** | C18 | 1.20x |
| GCP c4a-highcpu-32 | 0.6B | C16 (wave only) | **C16** | C18 | 1.20x |

Both customer reports were regenerated as version 2 from this evidence and are retained
privately outside the repository.

## x86 / Turin Zen5: the cohort loses there too, but the serving case does not justify a change

`X86-COHORT-1` run 2026-09-15 on a fresh AWS EPYC 9R45 (Zen5 Turin, 32 cores, SMT off, 4 CCX
x 32 MiB L3, AVX-512 VNNI+BF16, **no AMX**), OSS 0.6B/1.7B checkpoints, English bank, `ryan`,
`SIMD=avx512bf16`, `--self-test` 0 failures, `check-isa` PASS.

**Microbench — the prediction was right.** Same three cells, `taskset -c 0-7`, profile env:

| cell | Turin x86 VNNI | Graviton4 | Axion |
|---|---:|---:|---:|
| 1 singleton (chunk 4) | 24.1 ms | 40.6 ms | 34.6 ms |
| 2 sequential singletons | 47.6 ms | 81.0 ms | 70.1 ms |
| 1 ragged cohort | **65.0 ms** | 132.1 ms | 84.3 ms |
| penalty | **1.37x** | 1.63x | 1.20x |

chunk 8 confirms: `41.8 / 83.1 / 117.4 ms` = **1.41x**, and the sequential pair is again exactly
linear (2 x 24.1 = 48.2 against 47.6 measured). **CONFIRMED on a third host and a second ISA:
the ragged cohort is a per-call loss everywhere it has been measured.** The cost is the shared
source structure of the multi kernel, not the instruction set — which is what the x86 twin's
identical `acc[3][4][2]` / runtime-`S` shape predicted.

**Serving — and this is why nothing was changed.** Paired screens, one variable, cap raised
4 -> 8 on every arm so admission is not fail-fast limited:

| arm | C | STREAM p95 | stall@250 | verdict |
|---|---:|---:|---:|---|
| 1.7B cohort ON | 12 | 0.903 | 1.03% | KNEE |
| 1.7B cohort OFF | 12 | 0.923 | **0%** | HEALTHY |
| 1.7B cohort OFF | 14 | 0.967 | 0.95% | HEALTHY |
| 1.7B cohort OFF | 16 | 0.991 | 1.45% | KNEE |
| 0.6B cohort ON | 20 | 0.973 | 3.55% | KNEE |

Unlike Arm, the x86 result is **mixed**: retiring the cohort takes stall@250 from 1.03% to zero
at 1.7B C12 but makes STREAM p95 slightly *worse* (0.903 -> 0.923). Nothing on the ladder
reached CLEAR; every point sits between 0.90 and 0.99, against 0.83 for the Arm points that
were qualified. A 30-minute gate here would have roughly 3% of realtime margin instead of 17%.

**DECISION (owner's call, 2026-09-15): stop, and leave x86 alone.** `turin-c8a-32c-vnni-product`
keeps `QWEN_SD_MULTISLOT=2` and its `C12 / C20` recommendations. The reasoning is sound and is
recorded so it is not re-opened by accident:

* the previously reported x86 operating points were produced with the **customer checkpoints and
  a mixed qualification workload**. This campaign used OSS checkpoints and an English bank, so
  these screens say where the knee is **for this workload**; they are not evidence that the
  earlier x86 recommendations were wrong, and must never be quoted as such;
* the measurable serving gain on x86 is small and mixed, so the change does not pay for the
  instance time to qualify it, let alone for disturbing a shipped product profile;
* the microbench result is nevertheless real and is now documented. It stays on the books as a
  known, unexploited inefficiency: **if the exact `S == 2` named-accumulator kernel is ever
  written, x86 is the host with the most per-call headroom left to reclaim (1.37-1.41x)**, and
  that is the point at which to revisit the profile — not before.

`X86-COHORT-2` (re-walking the Turin ladder) is therefore **closed unstarted**: the ladder was
walked far enough to see the knee, and the answer did not justify a qualification.
