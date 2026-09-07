# P4 ragged panel claim-first allocation — 2026-09-07

## Task · Question

Test the smallest reviewer-directed reduction of ragged decoder scratch churn: claim a
real panel before allocating the worker's temporary im2col/INT8 activation buffers, then
reuse those buffers while the worker drains further panels. Does this reduce short-job
resource work without changing panel arithmetic or serving behavior?

## Known facts

- The change is limited to `sd_rag_panel_worker` in `qwen_tts_speech_decoder.c`.
- The candidate keeps the existing panel cursor, `qwen_sd_pool_run` ownership, AMX Design-D
  path, output ownership, fallback and allocation-failure semantics.
- A dispatched ragged job uses six logical `qwen_parallel` callbacks at the tested
  six-thread worker setting; a one-panel job stays on the caller because the threshold is
  two panels. The old default allocated before claiming, so dispatched short jobs could
  allocate in callbacks that later found no panel.

## Unknowns

- The harness does not expose allocator-only wall time or a perfect historical allocation
  counter. The pre-claim allocation count and bytes below are derived from the old source
  control and the candidate's per-call scratch sizes, not measured from an old counter.
- The short A/B cannot establish a durable serving effect for this small resource change;
  longer allocator-specific evidence is intentionally out of scope.

## Files/functions inspected

`qwen_tts_speech_decoder.c`: `sd_rag_panel_worker`, `rag_conv1d_amx`;
`qwen_tts_kernels.c`: `sd_pool_run`, `qwen_parallel`;
`tests/serve_parallel_wave.py`; commits `7294519`, `3a0ed6c`, `f5354b6`.

## Evidence

The candidate was built clean with `SIMD=amx` from source fingerprint `f5354b6:clean` on
the canonical c4-standard-24 AMX reference: twelve physical CPUs online, one socket/NUMA
domain and SMT off. Its binary SHA-256 prefix was `f892c6bf45eba030`; the old pre-claim
control binary from `7294519:clean` was `283e74b04600d75e`. Both passed the in-process
AMX capability table, self-test and decoder dispatch gate before serving.

The KPI A/B used the same 1.7B model, short diverse bank, true simultaneous waves,
`2x6`, batch cap 2, engine pool, q8, Design-D INT8, warm strip, ragged threshold 2,
three waves at C3 and C4, no profiler/census and zero coalesced reads. All 21 requests
per arm completed with zero errors, rejects and timeouts.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall@250 / @500 | req/s | B | cores |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| pre-claim control | 3 | 158 / 165 | 0.682 / 0.794 | 0.712 / 0.847 | 153 / 283 | 295 / 417 | 494 | 0% / 0% | 1.39 | 1.77 | 8.1 |
| claim-first | 3 | 158 / 170 | 0.694 / 0.809 | 0.725 / 0.862 | 153 / 283 | 315 / 451 | 500 | 0% / 0% | 1.36 | 1.77 | 8.0 |
| pre-claim control | 4 | 168 / 173 | 0.765 / 0.877 | 0.832 / 0.928 | 214 / 357 | 376 / 521 | 508 | 17% / 0% | 1.74 | 2.38 | 9.2 |
| claim-first | 4 | 175 / 176 | 0.764 / 0.817 | 0.836 / 0.893 | 172 / 300 | 347 / 476 | 516 | 17% / 0% | 1.70 | 2.31 | 8.8 |

The C4 STREAM p95 is lower for the candidate in this sequence, but C3 moves slightly
up, throughput is slightly lower and the change is small relative to short-wave run
variation. This is not sufficient to claim a serving-capacity improvement or to start
another tuning loop.

A separate one-wave C4 diagnostic enabled `QWEN_SD_RAG_STATS=1`; its overhead is not a
KPI result. Across 132 `[SDRAG]` records it observed 4,392 panels and 618 claim-first
scratch allocations, with zero fallback panels. The allocation shape was:

| panels in call | records | claim-first allocations per record |
|---:|---:|---:|
| 1 | 18 | 1 |
| 2 | 12 | 2 |
| 3 | 6 | 3 |
| 4 | 3 | 4 |
| 5 | 12 | 5 |
| 10 / 20 / 30 / 40 / 60 / 80 / 120 / 240 | 81 | 6 |

For the same records, the old source implies 702 pre-claim allocations: one for each
one-panel caller execution and six for each dispatched call. Candidate scratch total was
732.408 MB; scaling each dispatched record to six allocations gives an estimated
971.060 MB pre-claim total, or 12.0% fewer allocation events and 24.6% less temporary
scratch material. This proves the intended best-effort race reduction, not a measured
wall-time reduction.

## Conclusion

**RETAIN, NOT PROMOTE AS A SERVING WIN.** Claim-first is a bounded, default-safe resource
hygiene change: it demonstrably avoids allocating for workers that never claim a panel,
while preserving the existing complete fallback on allocation failure. The clean C3/C4
A/B is KPI-neutral within evidence and does not justify changing the serving policy or
claiming that decoder intercept has been solved.

## Next action

Do not reopen scratch reuse, claim-first tuning, ragged thresholds or AMX kernel work.
Keep P4 structural decoder cost open only for a separately justified intercept/rendezvous
target; a further change must first identify a measured dominant cost and preserve the
same fallback/parity contract.
