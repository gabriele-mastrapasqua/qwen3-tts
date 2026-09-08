# F3 — cross-worker cohort coincidence (2026-09-08)

Status · closed; no batching implementation.

## 1. Question

Do the two cap-2 prefork workers naturally reach Talker/CP-ready iterations close
enough in time that a global cohort could regularly form useful merged B=3/B=4
work without deliberately waiting for width?

Verdict · **NOT JUSTIFIED ON THIS HOST.** In the measured C4 steady state, only
`2.7%` of ready-event slots participate in a useful B>=3 opportunity within
`±1 ms`, `3.6%` within `±2 ms`, and `11.7%` even within `±8 ms`. The result is
well below the `25%` lower bound for preserving global batching as a near-term
server direction. This does not measure a batching speedup or prove that a
deliberate wait could never help; it says that natural cross-worker coincidence
is too sparse on this 2x6 host.

## 2. Ground truth

| item | value |
|---|---|
| source | `56e0dc1:clean` |
| branch | `feature/x86-amx-vnni-oss` |
| host | GCP `c4-standard-24`, Intel Xeon Platinum 8581C |
| topology | one socket/NUMA; physical CPUs `0-11`; SMT off; workers `0-5` and `6-11` |
| build | clean `SIMD=amx`; binary SHA-256 `0898de9bb83a7236a0dd361d49daf1ac04b4cf4c0b8ba9f33e3c915d9fbff9fd` |
| runtime | 1.7B INT8; Design-D INT8; fused residual on; q4; ragged threshold 2; engine pool; synchronous output; cap 2 |
| trace run | `TRUE_SIMULTANEOUS_WAVE`, C4, 3 waves, `tests/load_texts_en.txt`, seed base 42 |
| diagnostics | existing `[ITER]` trace only; no profiler or census |

The shipped source snapshot has no `.git` directory on the box, so the harness's
generic `dirty=yes` field is not authoritative; the embedded source fingerprint
and local HEAD identify the clean source. No benchmark or server process survived
the run; SMT remained off with CPUs `0-11` online.

The run's client anchor was healthy enough for a trace-only diagnostic: TTFB
p50/p95 `60/122 ms`, TTFA `429/594 ms`, STREAM_RTF `0.790/0.805`, required
prebuffer `134/163 ms`, safe-play-start `567/702 ms`, max-gap `287/293 ms`,
stall@100 `16.7%`, stall@250 and @500 `0%`, zero errors/rejects, and receive
coalescing `0%`. These are context, not a new qualification claim.

## 3. Trace point and population

The existing default-off `[ITER]` line is emitted at the beginning of each
continuous engine loop iteration, immediately before the frame-loop work is
entered. It carries:

* `CLOCK_MONOTONIC` timestamp;
* PID, iteration sequence and `n_active`;
* `n_active` as the current worker-side B proxy.

`[TOPOLOGY]` maps each PID to prefork worker and CPU mask. No kernel or scheduler
instrumentation was added, and no cross-worker synchronization was introduced.
This is an iteration/Talker-ready boundary proxy, not a per-kernel timestamp.

The harness first performs four warm-up requests; those trace events were excluded.
The analyzed population is the 12 real requests from three C4 waves (seeds
`4042`–`4053`). For each wave, startup/admission is the interval from the first
engine admission to the latest first PCM event. Steady state begins after all four
requests in that wave have produced first PCM and runs until the next wave's first
admission, or until the trace ends for the final wave. Events with `n_active=0`
are excluded from ready-work counts.

This classification intentionally measures only ready iterations. Prefill time
before an iteration is visible remains outside the startup event count.

## 4. Event population

| phase | wave 1 | wave 2 | wave 3 | total | worker 0 / worker 1 | B1 / B2 |
|---|---:|---:|---:|---:|---:|---:|
| startup/admission | 4 | 2 | 4 | 10 | 4 / 6 | 0 / 10 |
| steady state | 307 | 409 | 396 | 1112 | 625 / 487 | 731 / 381 |

The steady-state trace therefore contains mostly B1-ready iterations, with 381
B2-ready iterations. A cross-worker merged B>=3 opportunity requires pairing a
B2 event with a B1/B2 event; B1+B1 only produces hypothetical B2.

## 5. Offline matching method

For each window, candidate cross-worker pairs were formed from events in the same
phase. Candidate pairs were sorted by absolute timestamp distance and greedily
matched so that each event participates at most once. This gives a conservative,
non-duplicated hypothetical cohort count. Separately, `event coverage` counts an
event if any event from the other worker exists within the window, even if that
counterpart is also near another event.

`B2`, `B3`, and `B4` below mean the hypothetical sum of the two matched worker
values: B1+B1, B1+B2, and B2+B2 respectively. `useful event fraction` is
`2 * matched(B3+B4) / all steady ready events`; `useful pair fraction` is the
same useful count divided by all matched pairs.

## 6. Coincidence histogram

| window | startup events / matched pairs | startup useful event fraction | steady events | steady event coverage | steady matched pairs | B2 / B3 / B4 pairs | steady useful event fraction | steady useful pair fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ±0.25 ms | 10 / 0 | 0.0% | 1112 | 1.62% | 9 | 0 / 5 / 4 | 1.62% | 100.0% |
| ±0.5 ms | 10 / 0 | 0.0% | 1112 | 1.80% | 10 | 1 / 5 / 4 | 1.62% | 90.0% |
| ±1 ms | 10 / 0 | 0.0% | 1112 | 2.88% | 16 | 1 / 11 / 4 | 2.70% | 93.8% |
| ±2 ms | 10 / 0 | 0.0% | 1112 | 4.14% | 23 | 3 / 13 / 7 | 3.60% | 87.0% |
| ±4 ms | 10 / 0 | 0.0% | 1112 | 7.91% | 44 | 7 / 24 / 13 | 6.65% | 84.1% |
| ±8 ms | 10 / 0 | 0.0% | 1112 | 14.75% | 82 | 17 / 47 / 18 | 11.69% | 79.3% |

Startup has only ten logged ready events because the workers spend much of the
admission/prefill interval before the continuous iteration boundary is reached;
the startup zero is therefore a measured trace fact, not a claim that prefill
itself has no overlap. The decision is driven by the much larger steady-state
population.

## 7. Measured versus derived versus unknown

Measured: host/topology state, clean AMX build identity, client anchor metrics,
`[TOPOLOGY]` worker mapping, `[ITER]` timestamps, `n_active`, event counts, and
zero errors/rejects/coalescing.

Derived: startup/steady interval boundaries from per-request admission and first
PCM timestamps; warm-up exclusion; event-side coverage; non-overlapping pair
counts; hypothetical merged B distribution and useful fractions.

Unknown: actual cross-worker state migration cost, KV/state ownership feasibility,
cache and memory-bandwidth effects, AMX speedup for a merged B3/B4 operation,
whether an intentional wait could pay back its cadence cost, and decoder behavior
after global consolidation. F3 does not establish any of these.

## 8. Architecture implication

On this 2x6 host, natural worker readiness is too asynchronous for a global
Talker/CP cohort to be a high-probability source of B>=3 work without waiting.
At the strict `±1–2 ms` windows relevant to streaming cadence, useful coverage is
only `2.7–3.6%`; even the permissive `±8 ms` window reaches only `11.7%`.

Therefore preserve global batching only as a larger-host or future scheduling
research direction. Do not implement cross-worker state consolidation, deliberate
batch waits, or a new batching design from this evidence.

## 9. Next action

F3 is complete. The next falsifier should be a separately authorized admission/
lead-isolation experiment (for example, whether utilization-aware admission can
protect the cap-2 C4 envelope); it is not executed or authorized by this note.
