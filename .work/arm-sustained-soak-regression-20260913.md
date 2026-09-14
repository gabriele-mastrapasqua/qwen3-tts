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
