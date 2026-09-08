# F-cap3 — C5 capacity screen (2026-09-08)

Status · closed; cap 3 is not promoted.

Question · Does permitting a third request slot per prefork worker make C5
interactive on the 12-physical-core AMX host without materially harming the
already-running streams?

Verdict · Cap 3 removes the multi-second listener-backlog tail, but it does not
provide an economical realtime C5 envelope. It accepted all tested requests, yet
the aggregate `STREAM_RTF` p95 was `0.969` and the fifth-request proxy reached
`1.028`, with `stall@250ms = 13.3%`. Keep cap 2/q4 as the reference; do not
promote cap 3 or start C6 from this result.

## 1. Ground truth

| item | value |
|---|---|
| branch | `feature/x86-amx-vnni-oss` |
| source | `f07b0d2:clean` |
| latest checkpoint | `f07b0d2` (`docs: record F2 C5 startup decomposition`) |
| host | GCP `c4-standard-24`, Intel Xeon Platinum 8581C |
| topology | one socket/NUMA, CPUs `0-11` online, SMT off |
| build | clean `SIMD=amx`; binary SHA-256 `37b2be5d34999fb8ff4fc239d511f87b3761dcc614d2832891fe4d7bc1c54a17` |
| runtime | 1.7B INT8, Design-D INT8, fused residual on, q4, ragged threshold 2, engine decoder pool, synchronous output |
| prefork | 2 workers × 6 CPUs; no surviving benchmark/server process after either arm |
| diagnostics | `QWEN_TTFA_TRACE=1` and life trace only; no profiler/census in KPI runs |
| transport | client receive coalescing `0%` in both arms |

The F1 evidence anchor was `9aecbb3`; the current source includes the F2
diagnostic instrumentation but no runtime behavior change after that checkpoint.
Both arms used the same true-simultaneous mixed bank, three waves, seed discipline,
model, topology and flags. The harness `--batch-cap` is the current explicit
capacity control and is coupled to both parent connection cap and child engine
batch width; this coupling is part of the result, not an independent hidden knob.

## 2. Code-path audit

At this HEAD the parent creates the listener with backlog 16. The prefork loop
polls the listener only while a worker has `active[w] < cap`, unless the explicit
full-cap fail-fast mode is enabled. After `accept()`, it chooses the least-loaded
worker with capacity, sends the fd over the worker socketpair, and increments the
worker's whole-connection counter. The child parses the request and queues it for
the engine; the child grace queue timeout starts only after that point.

`cap = max_batch` in the prefork server, and the same `max_batch` becomes the
batched engine slot limit. Thus cap 3 tests two coupled facts at once:

* up to three whole connections may be assigned to one worker; and
* the worker can expose `B=3` to the Talker/CP/decoder loop.

This is the smallest existing production control and answers the requested
mechanical-capacity question, but it is not a parent-only admission experiment.
No listener backlog wait was visible in the cap-3 arm; the residual fifth-request
wait moved into child/engine admission.

## 3. Experiment arms

| arm | concurrency/cap | configuration |
|---|---:|---|
| control | C4 / cap 2 | fused + q4, Design-D INT8, 2x6, engine pool |
| treatment | C5 / cap 3 | identical settings, cap 3 only |

The workload launched five requests per wave rather than establishing four
streams and then admitting a fifth. Therefore “first four” below is a launch-index
proxy, not a causal established-stream population. This limitation prevents a
strong claim about the exact damage to four already-playing streams, but does not
hide the aggregate cap-3 failure.

## 4. Aggregate client and playback metrics

Values are p50/p95; latency and playback values are milliseconds unless noted.
Percentiles contain accepted requests only. Margins are `1 - STREAM_RTF`.

| metric | C4 cap 2 control (12 accepted) | C5 cap 3 (15 accepted) |
|---|---:|---:|
| TTFB | 54.8 / 118.1 | 76.0 / 449.0 |
| TTFA | 426.9 / 539.3 | 556.6 / 612.7 |
| STREAM_RTF | 0.820 / 0.835 | 0.831 / 0.969 |
| stream margin | 0.180 / 0.165 | 0.169 / 0.031 |
| TOTAL_RTF | 0.852 / 0.977 | 0.864 / 1.257 |
| required prebuffer | 196.0 / 490.1 | 244.2 / 366.3 |
| safe play start | 612.2 / 1029.4 | 816.8 / 979.9 |
| max gap | 333.3 / 504.3 | 325.0 / 563.0 |
| stall max | 125.6 / 424.3 | 150.7 / 190.3 |
| stall rate @100 ms | 50.0% | 86.7% |
| stall rate @250 ms | 0% | 13.3% |
| stall rate @500 ms | 0% | 0% |
| stall rate @1000 ms | 0% | 0% |
| total stall p95 @100 ms | 57.5 ms | 207.0 ms |
| total stall p95 @250/@500/@1000 | 0 / 0 / 0 ms | 43.0 / 0 / 0 ms |
| throughput | 0.254 req/s | 0.296 req/s |
| measured core-equivalent | 7.10 | 8.04 |
| context switches | 5,416/s | 5,889/s |
| PSS | 8.6 GiB | 8.8 GiB |
| errors | 0 | 0 |
| rejects | 0 | 0 |
| execution timeouts | 0 | 0 |
| receive coalescing | 0% | 0% |

The C4 and C5 throughput numbers are not a capacity comparison at equal
concurrency. The relevant result is that cap 3 adds work and admission pressure
while leaving only about 3.1% aggregate stream margin.

## 5. New fifth request versus four-request proxy

The three launch-index fifth requests in the C5 arm had:

| population | count | TTFB p50/p95 | TTFA p50/p95 | STREAM p50/p95 | TOTAL p50/p95 | prebuffer p50/p95 | safe start p50/p95 | max gap p50/p95 | stall@100/@250/@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first-four launch proxy | 12 | 76.0 / 185.1 | 558.0 / 612.7 | .811 / .862 | .855 / .921 | 245.3 / 366.3 | 819.5 / 979.9 | 352.4 / 563.0 | 83.3% / 16.7% / 0% |
| fifth launch proxy | 3 | 437.8 / 485.8 | 556.6 / 606.6 | .969 / 1.028 | 1.257 / 1.313 | 244.2 / 324.4 | 816.8 / 850.8 | 325.0 / 330.3 | 100% / 0% / 0% |

All three fifth-proxy connections were accepted by the parent without the
multi-second F2 backlog delay. Their detailed path samples were:

| sample | parent accept → child | enqueue → engine admit | engine admit → first PCM | client TTFB | client TTFA | STREAM_RTF |
|---|---:|---:|---:|---:|---:|---:|
| fifth proxy 1 | 0.169 ms | 65.367 ms | 423.138 ms | 66.0 ms | 492.4 ms | 1.028 |
| fifth proxy 2 | 0.079 ms | 436.959 ms | 118.763 ms | 437.8 ms | 556.6 ms | 0.904 |
| fifth proxy 3 | 0.770 ms | 484.528 ms | 119.877 ms | 485.8 ms | 606.6 ms | 0.969 |

So the answer to “does it become interactive?” is qualified: cap 3 removes the
seconds-long pre-accept wait and keeps fifth-proxy TTFA below 700 ms, but it does
not guarantee immediate engine admission and does not keep that population safely
realtime.

There is no clean established-four-before-fifth arm in this run. The four-request
proxy has no @500 ms stalls, but it has 16.7% @250 ms stalls and a 612.7 ms TTFA
p95. Compared with the cap-2 C4 control it is worse in TTFA, stream p95,
safe-start p95 and @250 ms stalls; request-bank and wave timing differ, so this is
indicative rather than causal. A causal established-stream degradation value is
UNKNOWN from this experiment.

## 6. Effective B and worker evidence

The main trace interval excludes the warm-up region. It recorded these active-slot
occupancies:

| run | n_active=0 | n_active=1 | n_active=2 | n_active=3 |
|---|---:|---:|---:|---:|
| C4 cap 2 | 6 | 731 | 389 | — |
| C5 cap 3 | 6 | 792 | 464 | 69 |

Worker assignment and effective batch were:

| run/worker | assigned/completed | effective B | measured cores |
|---|---:|---:|---:|
| C4 cap 2 / worker 0 | 6 / 6 | 0.84 | 3.13 |
| C4 cap 2 / worker 1 | 6 / 6 | 1.09 | 3.98 |
| C5 cap 3 / worker 0 | 9 / 9 | 1.57 | 5.03 |
| C5 cap 3 / worker 1 | 6 / 6 | 0.85 | 3.01 |

Per-worker maximum active-slot counts were not emitted separately; the run-wide
trace proves B=3 was reached, while the parent assignment proves worker 0 absorbed
the uneven 9/6 connection split. Per-worker B3 exposure beyond that is UNKNOWN.

The decoder groups provide direct B3 evidence without a profiler:

| run | decoder group | calls | duration median/p95 |
|---|---|---:|---:|
| C4 cap 2 | B1 | 184 | 59.125 / 62.619 ms |
| C4 cap 2 | B2 | 106 | 88.986 / 92.770 ms |
| C5 cap 3 | B1 | 199 |  — |
| C5 cap 3 | B2 | 121 |  — |
| C5 cap 3 | B3 | 21 | 117.417 / 125.472 ms (max 148.518) |

The approximate interval between `[ITER]` records, not an isolated step benchmark,
was:

| run | B | iteration wall median/p95 |
|---|---:|---:|
| C4 cap 2 | 1 | 38.739 / 98.189 ms |
| C4 cap 2 | 2 | 40.825 / 147.029 ms |
| C5 cap 3 | 1 | 39.813 / 102.123 ms |
| C5 cap 3 | 2 | 40.559 / 137.852 ms |
| C5 cap 3 | 3 | 40.248 / 197.642 ms |

Trace logging is present in these diagnostic runs, so the interval is evidence of
loop behavior, not a clean B-specific cost model. The direct B3 decoder duration
and the 69 B3 iterations prove that B3 work occurred; they do not by themselves
prove that all worker time was saturated by B3.

## 7. AMX and correctness evidence

The clean binary reported `SIMD=amx`, active AMX INT8/BF16 capabilities, and passed
the self-test and dispatch-map gates. Startup reported 24 persistent INT8 decoder B
packs (19.3 MB). The cap-3 startup log observed real AMX INT8 use at B=3 in the
in-region runner. KPI runs deliberately had no census, so this note does not claim
a per-run TDPBSSD tile count or a new AMX wall-share percentage. The Design-D
path, flags and persistent packs were unchanged from the qualified reference.

No code or numerical behavior was changed for F-cap3. Both arms completed with zero
errors, rejects and execution timeouts; receive coalescing was zero. No new WAV
parity run was required because this was a capacity-only experiment with identical
runtime code and control settings.

## 8. Promotion decision

| criterion | result |
|---|---|
| all C5 requests accepted without multi-second parent backlog | PASS in this short arm |
| fifth request interactive by TTFA ≤700 ms | PASS for 3/3 proxy samples, with 437.8–485.8 ms TTFB and 492.4–606.6 ms TTFA |
| aggregate STREAM_RTF p95 ≤0.95 | FAIL: 0.969 |
| fifth-proxy STREAM_RTF p95 ≤0.95 | FAIL: 1.028 |
| stall@500 ≤5% | PASS: 0% |
| useful @250 continuity | FAIL: 13.3% overall; 16.7% first-four proxy |
| established-four causal non-degradation proven | UNKNOWN; wave design was not sequential |
| errors/rejects/timeouts/coalescing | PASS: all zero |

F-cap3 is therefore **FAIL / not promoted** for the target. A third slot is
mechanically accepted by the server and removes the hidden listener wait, but the
12-core host does not have enough measured margin to serve C5 safely under this
configuration. The failure is not a correctness failure; it is a realtime/capacity
failure with a secondary engine-admission queue term.

## 9. Measured, derived and unknown

Measured directly or emitted by the harness: client KPIs, playback simulations,
HTTP outcomes, coalescing, worker assignment, `[ITER]` occupancy, decoder group
durations, and clean-build/AMX gates.

Derived: stream margin, population splits by launch index, and the interpretation
that the cap-3 arm removed the F2 listener-backlog term. Userspace cannot observe a
pre-`accept()` packet timestamp directly; in this arm the absence of a delayed
accept was established from client/server trace alignment, not packet tracing.

Unknown: causal impact on four already-established streams, a clean isolated B3
Talker/CP step cost, per-worker B3 distribution, and any exact AMX request-wall
share in the KPI runs.

## 10. Next action

Keep cap 2/q4 as the C4 reference. Do not promote cap 3, do not run C6, and do not
infer that immediate 503 or a third slot increases capacity. The next capacity or
admission change requires a separately authorized experiment; F-cap3 itself is
closed here.
