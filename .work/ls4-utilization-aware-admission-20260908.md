# LS-4 utilization-aware admission falsifier — 2026-09-08

Status · **closed; FAIL on the 12-physical-core AMX reference**.

Question · Can the parent keep a nominal cap of two, admit one occasional third
request when a worker has recent iteration headroom, and preserve the playback
envelope of four established streams?

Verdict · The fifth request became interactive in all three predeclared threshold
arms, but every treatment materially damaged the established four. The mechanism
is therefore not promoted and does not deserve a longer qualification campaign on
this host. Keep cap2/q4 fail-fast as the reference; freeze this admission idea
here and move the next comparison to another hardware/serving envelope rather
than tuning more thresholds locally.

## 1. Ground truth and provenance

| item | value |
|---|---|
| final source checkpoint | `85579a8` (`perf: add bounded utilization-aware admission probe`) |
| branch | `feature/x86-amx-vnni-oss` |
| host | GCP `c4-standard-24`, Intel Xeon Platinum 8581C |
| topology | one socket/NUMA; CPUs `0-11` online; SMT `control=off`, `active=0` |
| runtime reference | 1.7B INT8, Design-D INT8, fused residual on, q4, ragged threshold 2, engine-owned decoder pool, synchronous output |
| prefork | 2 workers × 6 threads; parent cap 2; `--max-queue 0` for immediate full-cap decisions |
| KPI binary | AMX SHA-256 `6a9193aee70dc77591b84f466fc96bbc4f514906c767628bd7af7b6ea2abc475`; source `3b901fe-dirty:d0a78ba37964` |
| final rebuild | clean source `85579a8:clean`; AMX SHA-256 `3a3100fded50aebc166173a9b5a78b91376258770e4aaf7fc6ea075d5311e92c`; self-test passed |
| KPI diagnostics | `QWEN_TTFA_TRACE=1`, `QWEN_LIFE_TRACE=1`, plus `[ADMITUTIL]` only for treatment; no profiler or census |
| client fidelity | coalesced reads: `0/834` established chunks and `0/207` fifth-request chunks in every treatment/control population |

The KPI binary predates only the final defensive `cap == 2` guard and documentation
line in `85579a8`; all measured arms used cap 2, so that guard is not on a measured
alternative path. The final committed tree was rebuilt cleanly and self-tested
after the experiment.

## 2. Admission code-path audit

The parent owns `active[w]`, the number of whole connections dispatched to worker
`w`. Normal selection chooses the least-loaded worker with `active[w] < 2`. The
listener remains polled while a normal slot exists; LS-4 also keeps it polled when
the diagnostic policy is enabled so a full-cap arrival is decided immediately.

When all normal slots are full, LS-4:

1. refuses to use a second temporary slot if any worker already has `active[w] > 2`;
2. reads one per-worker health record from a `MAP_SHARED` page inherited across fork;
3. accepts a worker only when its sample has a nonzero sequence, age no greater than
   `max(2 * limit_ms, 100 ms)`, and `0 < last_iter_ms < limit_ms`;
4. selects the eligible full worker with the smallest recent interval and dispatches
   one third connection;
5. otherwise closes the accepted connection with the existing immediate overload
   response (503), with no listener-backlog wait and no polling loop.

The child is provisioned with `B=3` only while `QWEN_ADMIT_UTIL=1`; the parent
still admits at most two normally and allows only one global temporary extra
connection. There is no migration, preemption, work stealing, state consolidation
or deliberate batch wait. Once dispatched, the third request follows the existing
engine path.

The health signal is the existing continuous-loop start-to-start interval published
at the loop boundary. It is a conservative service-loop proxy, not a pure kernel
wall measurement and not aggregate CPU utilization. The policy is Linux-prefork
only and default-off.

## 3. Workload and fixed thresholds

The diagnostic harness `tests/ls4_admission_probe.py` ran three repetitions per arm
from the long rows of the stable English bank. Four requests were started together;
the fifth was started only after all four had returned their first PCM chunk and an
additional 500 ms settle interval had elapsed. Requests were then allowed to finish.
The fifth request used the next long row and a deterministic seed offset.

Thresholds were fixed before running and were not tuned after results:

| arm | policy | iteration limit |
|---|---|---:|
| control | cap2, fail-fast, no LS-4 | — |
| conservative | transient extra slot | 40 ms |
| medium | transient extra slot | 60 ms |
| permissive | transient extra slot | 80 ms |

The server-side F2 parent trace is authoritative for overload decisions. In the
control, one fifth was observed as HTTP 503 and two clients saw `BrokenPipe` while
the parent closed the already-rejected socket; the parent log records all three
full-cap rejects. This is an expected fail-fast transport observation, not an
inference error or an accepted-request server failure.

## 4. Acceptance and real B3 evidence

| arm | fifth accepted | parent full-cap rejects | real `admit3` decisions | child `n_active=3` ITER lines | B3 batch observations |
|---|---:|---:|---:|---:|---:|
| cap2 control | 0/3 | 3 | 0 | 0 | 0 |
| limit 40 | 3/3 | 0 | 3 | 737 | 6 |
| limit 60 | 3/3 | 0 | 3 | 754 | 6 |
| limit 80 | 3/3 | 0 | 3 | 737 | 6 |

The B3 observations are the child `[ITER] n_active=3` records and the existing
`[BATCH] ... in-flight admitted=3` records. Thus treatment acceptance was not
merely a parent counter or a queued HTTP illusion.

At the three admission decisions, the selected worker's recent interval was:

| limit | recent interval at decision, ms | sample age, ms |
|---:|---|---|
| 40 | 37.172, 37.114, 37.272 | 88.896, 94.858, 91.269 |
| 60 | 36.575, 37.318, 37.661 | 26.916, 14.733, 52.428 |
| 80 | 37.016, 38.537, 37.572 | 97.239, 44.651, 50.244 |

Measured same-worker iteration intervals (start-to-start, all repetitions) were:

| limit | B1 p50/p95 ms | B2 p50/p95 ms | B3 p50/p95 ms |
|---:|---:|---:|---:|
| 40 | 41.4 / 104.1 | 43.8 / 138.3 | 46.0 / 178.1 |
| 60 | 41.5 / 102.9 | 43.6 / 142.3 | 45.5 / 174.9 |
| 80 | 41.2 / 102.4 | 43.7 / 152.3 | 46.0 / 174.4 |

The B3 p95 increase is not a sufficient safety proof by itself, but it is consistent
with the playback disturbance below. Core-equivalent utilization and context-switch
counts were not sampled in this targeted run and remain **UNKNOWN**.

## 5. Established-four disturbance

The table reports the twelve established requests per treatment/control. Percentiles
are nearest-rank p50/p95. `max gap` and prebuffer values are milliseconds; stalls are
the number of requests with at least one stall at that fixed buffer.

| arm | STREAM_RTF p50/p95 | TTFA p50/p95 | required prebuffer p50/p95 | safe play start p50/p95 | max gap p50/p95 | stall@250 | stall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap2 control | 0.817 / 0.835 | 707 / 794 | 134 / 165 | 829 / 936 | 319 / 391 | 0/12 | 0/12 |
| limit 40 | 0.834 / 0.993 | 710 / 781 | 210 / 659 | 990 / 1369 | 321 / 704 | 6/12 | 2/12 |
| limit 60 | 0.864 / 1.004 | 707 / 778 | 494 / 801 | 1200 / 1577 | 482 / 948 | 6/12 | 2/12 |
| limit 80 | 0.848 / 0.985 | 700 / 784 | 151 / 529 | 862 / 1241 | 368 / 657 | 6/12 | 0/12 |

The local gap window around the fifth request shows the same causal direction:

| arm | established max-gap before fifth p50/p95 | after fifth p50/p95 |
|---|---:|---:|
| cap2 control | 233 / 297 | 263 / 360 |
| limit 40 | 173 / 244 | 319 / 704 |
| limit 60 | 219 / 333 | 295 / 653 |
| limit 80 | 184 / 262 | 368 / 657 |

The treatment p95 after-admission gaps are roughly 2–3× the pre-arrival p95 and
the fixed-buffer stall rate at 250 ms is 50% in every treatment. This fails the
established-stream requirement even though the control population itself has no
fixed-buffer stalls in this long-text probe.

## 6. Fifth-request behavior

All nine treatment fifth requests were accepted. The table is p50/p95 across the
three fifth requests in each arm.

| arm | TTFB ms | TTFA ms | STREAM_RTF | required prebuffer ms | safe play start ms | max gap ms | stall@250 | stall@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| limit 40 | 39 / 102 | 394 / 463 | 0.965 / 0.966 | 315 / 345 | 705 / 807 | 355 / 396 | 0/3 | 0/3 |
| limit 60 | 113 / 114 | 479 / 480 | 0.964 / 0.976 | 293 / 540 | 772 / 1019 | 399 / 947 | 1/3 | 0/3 |
| limit 80 | 77 / 84 | 426 / 444 | 0.961 / 0.962 | 243 / 304 | 654 / 748 | 383 / 491 | 0/3 | 0/3 |

Server `[PATH]` timestamps show a bounded but nonzero child admission wait for
accepted fifth requests. The p50/p95 values were:

| arm | child `enqueued -> admitted`, ms | `admitted -> first PCM`, ms |
|---|---:|---:|
| limit 40 | 38 / 101 | 356 / 361 |
| limit 60 | 110 / 113 | 365 / 367 |
| limit 80 | 76 / 83 | 360 / 363 |

The fifth request is therefore individually interactive and has no multi-second
pre-accept tail, but it has little realtime margin itself and does not compensate
for the damage to established playback.

## 7. Measured, derived and unknown

Measured:

- topology/SMT state, binary/source identity, flags and clean final self-test;
- parent admission decisions and server reject counts;
- real child B3 occupancy and iteration intervals;
- client TTFB/TTFA/STREAM_RTF/playback simulation, fixed-buffer stalls and receive fidelity;
- child enqueue/admit/first-PCM timestamps for the accepted fifth requests.

Derived:

- local before/after gap windows relative to the fifth request's client start;
- child queue/admission and admitted-to-first-PCM durations from monotonic `[PATH]` fields;
- acceptance rate from the three deterministic fifth arrivals per treatment.

Unknown / not claimed:

- core-equivalent utilization and context-switch response;
- an exact causal split of each established stream's gap into decoder, Talker/CP,
  pool or transport work;
- long-soak tail behavior of the treatment;
- safe capacity on a different core count, NUMA layout or host;
- any global batching, migration, lead scheduler or overlap benefit.

## 8. Verdict and architecture implication

**FAIL on this host/reference.** The predicate admitted all three fifth arrivals for
each threshold. The fifth became interactive, but established STREAM_RTF p95 was
`0.985–1.004`, fixed-buffer stall@250 was `50%`, and the local post-admission max-gap
p95 reached `653–704 ms`. This fails both the preferred `<=0.90` and exploratory
`<=0.95` established-stream ceilings.

This is not a finding that every utilization-aware admission policy is impossible
on every machine. It is a falsifier of this minimal recent-iteration predicate on
the 12-core AMX reference. The result is also not a reason to tune more threshold
values: the three predeclared conservative/medium/permissive values all admitted
and all failed the established-stream gate.

Keep `QWEN_ADMIT_UTIL` default-off and keep cap2/q4 as the serving control. Do not
promote it or run a longer qualification on this host. The next action is
architecture freeze plus cross-host comparison of the already trusted cap2 envelope;
any future LS-4 reconsideration needs a materially different host or an independent
capacity signal, not another local threshold sweep.

## 9. Deferred risks

- The parent health page is intentionally a small diagnostic mechanism, not a general
  scheduler or production admission contract.
- The policy is scoped to the streaming/batched path measured here; other request kinds
  share the parent connection counter and were not part of this falsifier.
- The targeted harness does not sample CPU utilization or context switches; those remain
  required only if a later architecture decision needs them.
