# QL-1 C3 full qualification — GCP C4 highcpu-16 — 2026-09-08

Task · Complete the playback-aware QL-1 suite at the best 8-core AMX point and
decide whether the current serving architecture can be frozen.

Question · Does `1x8/C3` remain safe across input classes, long-request
admission, overload, slow clients and sustained operation?

Known facts · The host is an 8-physical-core Xeon Platinum 8581C with SMT off,
CPUs `0-7` online, one NUMA node and performance governor. The source and binary
are the clean AMX target used by the QL-2 slot. The current q4/fused/Design-D
configuration is the best 8-core topology, but QL-2 established C3 only on a
short-bank wave.

Unknowns · Legacy CLI goldens are not a server quality oracle. Per-class p95 in
the five-minute mixed SOAK is under-sampled. The long-arrival probe exercised an
accepted `2+1` admission rather than a `3+1` admission; `3+1` is correctly
rejected at cap 3.

Files/functions inspected · PLAN, `qwen_tts_server.c`, continuous serving and
decoder paths, WAVE/SOAK clients and analyzers, playback simulation, profile
preflight and the F1/F2/F-cap3/F3/QL-2 evidence.

Evidence · Raw artifacts remain outside the repository. The corrected SOAK used
the runner at `a4f3478`; accepted rows were re-analysed with the fixed analyzer
at `cfa9624`.

Conclusion · The runner is now operationally correct: fail-fast 503s are
separate admission outcomes, explicit profile topology/capacity arguments are
recorded, and under-sampled per-class tails are not called regressions. The
pooled C3 SOAK is stable, but the full QL-1 envelope is **NOT QUALIFIED**.
Long input and accepted long-prefill admission remain continuity blockers.

Next action · Keep cap2/q4 as the conservative cross-host reference until a
separately scoped prefill/lead architecture task addresses long-input coupling.
Do not call the 8-core C3 point production qualified from the pooled SOAK alone.

## 1. Architecture-freeze audit

| item | current state | QL-1 impact |
|---|---|---|
| bounded async output | implemented, default-off; small stopped-reader probe showed no induced disturbance | defer; no demonstrated blocker |
| SL-1 known-text layout | implemented, default-off; structural/current-generation gates pass | defer ICL/clone and quality/prefill scaling |
| LS-2 lead feedback | not implemented; LS-1 hard credit gate was falsified | defer; q4 remains fixed reference |
| PF-1 long-prefix prefill | not implemented | **open blocker for long-input startup/cadence** |
| decoder structural intercept | fused residual retained; full strip executor not implemented | defer; short/medium C3 path is usable |
| same-pool decoder overlap | rejected by P4 | closed for this generation |
| global Talker/CP batching | F3 coincidence too sparse | closed/deferred on this host |
| cap3 permanent / LS-4 temporary B3 | damaged realtime margin | rejected; no rescue tuning |
| ownership/topology rewrite | no evidence justifying it before QL-2 | defer |

`CURRENT SERVING ARCHITECTURE = QUALIFICATION-FROZEN` is **NO** for a full
production claim because long input and admitted long-prefill work can damage the
playback envelope. The kernel/dataflow reference is frozen for comparison;
changes require a new qualification generation.

## 2. Frozen candidate and revalidation

| field | value |
|---|---|
| host / CPU | GCP C4 highcpu-16 / Xeon Platinum 8581C |
| physical CPUs / mask | 8 / `0-7` |
| SMT / NUMA | off / one NUMA node |
| topology | one prefork worker × eight threads |
| admission | cap 3, `--max-queue 0`, queue timeout 0 |
| model | Qwen3-TTS 1.7B INT8 |
| decoder | Design-D INT8, ragged batch active |
| fused / warm strip | active / active |
| ragged threshold / quantum | 2 / q4 |
| pool / output | engine-owned decoder pool / synchronous |
| prefix cache | active |
| source | `a3e9ddd696a9a76d9ee7390aff1d13c9580863b9` clean |
| binary | AMX; SHA-256 `06b8cba62a2c5de62eaa9c3cec6b45ceeb8513cf65e5220c97d7e44e5e16051` |

Post-reboot CPU check, SMT/mask/governor/quota checks, caps, self-test, dispatch
map and strict AMX product preflight passed. The resolved leaves were ragged
Design-D INT8, fused residual and warm strip. No server or benchmark process
survived the final runs. Receive coalescing was zero in KPI arms.

The SOAK harness fixes in `e1e6d94`, `a4f3478` and `cfa9624` are test/evidence
changes only; they do not alter inference, scheduling, kernels or defaults.

## 3. Short / medium / long / mixed waves

Three-wave C3 screens used the same 1.7B model, bank, voice, seeds and AMX
preflight. Values are p50/p95 unless stated; times are milliseconds.

| class | n | TTFB | TTFA | STREAM_RTF | TOTAL_RTF | prebuffer | safe start | max gap p95 | stall @100/@250/@500/@1000 | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| short | 15 | 40.6/79.5 | 177.6/180.9 | .745/.778 | .809/.860 | 96/120 ms | 276/301 ms | 252 ms | 0/0/0/0 | PASS |
| medium | 15 | 90.7/181.6 | 293.0/340.5 | .749/.769 | .798/.819 | 94/128 ms | 387/427 ms | 269 ms | 0/0/0/0 | PASS |
| long | 15 | 249.8/515.8 | 855.4/870.4 | .795/.825 | .834/.865 | 113/145 ms | 969/1001 ms | 315 ms | 0/0/0/0 | FAIL: startup |
| mixed | 15 | 94.7/402.1 | 499.5/560.4 | .692/.785 | .750/1.000 | 124/160 ms | 603/673 ms | 344 ms | 20/0/0/0 | FAIL: tail margin |

Long streaming RTF is below one, but first-play latency scales with prefill and
exceeds the accepted continuity target. Mixed has acceptable TTFA/prebuffer but
its worst STREAM_RTF reaches the hard boundary and has a nonzero 100 ms stall
rate.

## 4. Corrected C3 SOAK

Run: five minutes plus 30 s warm-up, mixed bank, three closed-loop clients,
`1x8`, cap 3, q4, fused residual, Design-D, synchronous output, strict AMX
preflight, no profiler/census.

The first attempted run was invalid as a C3 run: before `a4f3478`,
`serve_soak.py` ignored explicit profile-mode `--batch-size 3` and topology
arguments, so the effective command was cap 2. It produced a 503 retry storm and
is excluded from KPI evidence.

The corrected run's pooled population and metrics were:

| metric | value |
|---|---:|
| completed / KPI samples | 94 / 94 |
| intentional 503 rejects | 9 |
| inference errors / queue timeouts / request timeouts | 0 / 0 / 0 |
| TTFB p50/p95 | 37.3 / 130.3 ms |
| TTFA p50/p95 | 198.2 / 453.2 ms |
| STREAM_RTF p50/p95 | 0.8657 / 0.9193 |
| required prebuffer p50/p95 | 175 / 445 ms |
| safe-play-start p50/p95 | 383 / 693 ms |
| max gap p95 | 614 ms |
| stall rate @100/@250/@500/@1000 | 81.9% / 8.5% / 1.1% / 0% |
| receive coalescing | 0% |
| resource stability | PASS; memory <1% drift, threads/fds stable |

All five pooled windows had STREAM_RTF p95 below one. The corrected analyzer
reports `SOAK RESULT: PASS` for pooled stability and marks each per-class result
`PARTIAL` because the five-minute run lacks enough samples for per-class p95.
`--strict-kpi` must still reject that incomplete per-class qualification. The
pooled SOAK is stability evidence, not a complete class-tail production gate.

## 5. Long-request arrival interference

The bounded probe admitted a long request while two established long streams were
already active. All requests completed and the new request had low initial
response latency, but established streams were measurably disturbed:

| quantity across 3 repetitions | established result |
|---|---:|
| max gap before injection | 188–229 ms |
| max gap after injection | 264–304 ms |
| injection-window max gap | 525–594 ms |
| stall @250 | present in 2/3 repetitions |
| stall @500 | 0 in all repetitions |
| admitted long request STREAM_RTF | about .731–.732 |

This is a real coupling signal, not a claim that every C3 request must fail: at
cap3 a fourth arrival is rejected. It does prove that an accepted long prefill
can consume established-stream cadence margin when a slot is available. PF-1 or
SL-1 remains a future architecture requirement; no change is made here.

## 6. Slow-client isolation

A stopped/very slow reader was run beside two normal long readers with synchronous
output. Both normal readers completed with approximately 829 ms TTFA, 0.293 s
maximum gap and identical output sizes; no unrelated-stream failure was observed.
This is a bounded PASS for the tested small population, not a broad concurrency
proof. Synchronous output is therefore not a demonstrated QL-1 blocker and async
output remains default-off.

## 7. 3+1 overload

Three established streams followed by a fourth request were run with cap3 and
fail-fast admission. All nine established requests returned 200; all three
fourth requests returned immediate 503. Established local max gaps were roughly
240–252 ms with no fixed-buffer stall at 250 or 500 ms. Intentional 503 is the
expected overload behavior, not an inference error.

Verdict: PASS for overload isolation.

## 8. Poisson/open arrivals

The open-arrival probe used the same AMX lane and separated accepted rows from
intentional rejects:

| offered rate | accepted / rejected | accepted STREAM_RTF p50/p95 | accepted TTFA p50/p95 |
|---:|---:|---:|---:|
| .15 | 15 / 0 | .748 / .948 | 183 / 496 ms |
| .25 | 12 / 3 | .779 / .860 | 168 / 363 ms |
| .35 | 9 / 6 | .746 / .894 | 164 / 414 ms |

Accepted requests remain realtime; overload transitions to prompt 503 rather than
hidden queue latency. This characterizes the envelope but does not turn C3 into a
full open-arrival qualification.

## 9. Quality, transport and error contract

The current-generation gate passed:

- source/binary/profile identity matched the AMX contract;
- resolved Design-D INT8, fused residual and q4 paths matched the lane;
- server emitted valid PCM/WAV output with valid duration/container;
- current fused on/off parity checks from the same generation remain valid;
- receive coalescing was zero in KPI runs;
- no inference errors, request timeouts, crashes or unexpected precision fallback.

`SEMANTIC QUALITY: NOT RE-QUALIFIED BY LEGACY CLI GOLDEN; SERVER
STRUCTURAL/PARITY GATES PASS.` The old CLI/librosa/0.6B golden workflow is not a
server QL-1 oracle and is excluded from this result.

Error/rejection taxonomy: HTTP 200 completed rows are valid; intentional 503
overload rows are admission outcomes; no timeout, connection, inference or
server-crash population was observed.

## 10. Final QL-1 decision

| field | result |
|---|---|
| highest short-bank GOOD concurrency | C3 on 1x8 |
| conservative full-envelope point | C2 pending a separately complete C2 suite |
| first failing full-envelope point | C3: long/mixed/accepted-long-arrival conditions |
| pooled C3 SOAK | PASS for stability; per-class tails PARTIAL |
| architecture qualification freeze | NO |
| C3 full QL-1 | FAIL / NOT QUALIFIED |
| output policy | synchronous; no demonstrated small-population slow-client blocker |
| quality | structural/current-generation PASS; legacy semantic oracle excluded |
| next QL-2 slot | AMD/Turin VNNI, then Axion/Arm |

The current reference is good enough for cross-host comparison at a clearly
stated C2/C3 operating envelope, but it must not be advertised as a fully
qualified C3 production point for long/mixed input until the long-prefill/admission
coupling is addressed by a separate architecture task.
