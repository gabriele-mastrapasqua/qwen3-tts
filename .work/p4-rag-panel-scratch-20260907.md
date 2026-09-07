# P4 ragged panel scratch reuse — 2026-09-07

## Task · Question

Test whether retaining the ragged decoder worker's temporary `col`, INT8 `colq` and
activation-scale buffers per pool pthread reduces the measured small-call/intercept
cost without changing the panel decomposition or AMX arithmetic.

## Known facts

- The candidate changed allocation lifetime only; it did not change tiling, panel size,
  pool threshold, quantisation, output order or the Design-D kernel.
- The control was the existing q8 / Design-D INT8 / warm-strip / ragged-threshold-2 /
  engine-pool configuration. The treatment enabled only
  `QWEN_SD_RAG_PANEL_SCRATCH=1`.

## Unknowns

- The short WAVE did not expose allocator-level timings or per-worker retained capacity;
  PSS was the available memory signal.
- The harness's `dirty=yes` field is not authoritative on the source-only remote copy:
  it cannot run `git diff` without `.git`; the binary embedded `7294519:clean` and the
  source fingerprint agreed.

## Files/functions inspected

`qwen_tts_speech_decoder.c`: `sd_rag_panel_worker`, `rag_conv1d_amx`;
`tests/serve_parallel_wave.py`; `docs/feature-flags.md`; `qwen_tts_kernels.c` flag
registry.

## Evidence

The candidate was committed as `7294519`, built clean with `SIMD=amx` on the canonical
GCP c4-standard-24 reference host, and passed `--caps`, `--self-test`, dispatch-map
generation, local self-test, flag registry, plan/integrity and diff checks. The host
had one socket/NUMA domain, CPUs 0–11 online and SMT off. The binary SHA-256 prefix was
`283e74b04600d75e` for both arms.

Both arms used the same 1.7B model, short diverse bank, true simultaneous waves,
`2x6`, batch cap 2, C3/C4, three waves, no profiler/census and zero coalesced reads.
All 21 requests per arm completed with zero errors, rejects and timeouts.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall@250 / @500 | req/s | effective B | cores |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| control | 3 | 159 / 167 | 0.672 / 0.846 | 0.703 / 0.910 | 174 / 331 | 335 / 489 | 545 | 11% / 0% | 1.37 | 1.77 | 8.0 |
| scratch | 3 | 158 / 168 | 0.680 / 0.826 | 0.712 / 0.887 | 185 / 294 | 347 / 454 | 497 | 0% / 0% | 1.36 | 1.77 | 8.0 |
| control | 4 | 168 / 173 | 0.766 / 0.810 | 0.839 / 0.885 | 168 / 344 | 340 / 509 | 534 | 17% / 0% | 1.71 | 2.33 | 8.6 |
| scratch | 4 | 169 / 172 | 0.772 / 0.843 | 0.850 / 0.893 | 184 / 420 | 353 / 581 | 540 | 17% / 0% | 1.70 | 2.35 | 8.9 |

The C3 treatment has a small lower p95 STREAM_RTF, but C4 moves from `0.810` to
`0.843` and required-prebuffer p95 from `344` to `420` ms. Throughput and effective
batch are unchanged within this WAVE. The result is not a stable server-level win;
therefore the candidate is not promoted and is removed from the active tree.

## Conclusion

**REJECT as a serving optimization; do not retain the flag or TLS cache.** The existing
per-task allocation path remains the control. This test does not prove that allocator
cost is zero; it proves that this uninstrumented per-worker retention scheme has no
measured C4 benefit and carries an avoidable lifetime/PSS risk.

## Next action

Keep P4 structural decoder-cost work open only for a separately measured fixed-cost
target. Do not reopen this scratch experiment or infer a kernel/dataflow conclusion
from its neutral result.
