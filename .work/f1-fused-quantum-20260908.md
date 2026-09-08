# Task · F1 fused residual × decoder quantum — 2026-09-08

Task · Execute the highest-information post-P4 falsifier: test whether the qualified
fused-residual decoder path makes a smaller complete decoder quantum preferable at C4.

Question · With all other serving controls fixed, does fused residual make q4 or q2 a
better playback/realtime point than q8, without a material TTFA or correctness cost?

Known facts · The source was frozen at HEAD `e60aebf65cd8d068c7ff89c2a5a0cd8e5511cd6a`
on `feature/x86-amx-vnni-oss`. The clean AMX build used the source archive SHA256
`b05b11203245a18f8f1deefea764c0832be782f40ffc85602d1039e0cd91dcb8` and produced
binary SHA256 `9d8052daf7e7b01fff1c0c642fae0308abbe4b48f5c8469cfcb08457bb7b9df4`.
The reference host was GCP c4-standard-24 / Xeon Platinum 8581C, one socket/NUMA,
12 physical CPUs online, SMT off, workers `0-5` and `6-11`.

Unknowns · This was a short three-wave KPI screen, not a five-minute qualification.
No per-arm profiler/census was enabled, so this note does not claim a new measured
TDPBSSD tile count, AMX MAC share, decoder wall share, or fused-vs-off causal delta.
The exact fused-off q8 anchor was not run because exact comparability with the archived
reference was not preserved. The earlier q2/q4/q8 control screen used a different
run/identity and already had q2/q4 below the preferred STREAM p95 target; it cannot be
used to attribute an F1 delta to fusion.

Files/functions inspected · `qwen_tts.c`, `qwen_tts_speech_decoder.c`, the serving
configuration and `tests/serve_parallel_wave.py`/playback simulation outputs. The
experiment used flags only; no source or runtime behavior was changed.

Evidence · The setup, paired KPI/playback tables and interpretation below summarize the
three final sequential arms. The raw benchmark JSON/JSONL and server logs remain in the
untracked evidence area; this tracked note intentionally records only public-safe
aggregates.

Conclusion · q4 is the next C4 playback/realtime reference candidate, q8 remains the
throughput control, and the fused-vs-off causal shift is not established by this screen.

## Setup

All arms used the same 1.7B INT8 model, English `ryan` voice, short diverse bank,
seed base 2027, true simultaneous C4 wave, concurrency 4, three waves, batch cap 2,
2×6 prefork, engine-owned decoder pool, synchronous output, warm strip, ragged
threshold 2, Design-D INT8 and fused residual enabled. The relevant explicit runtime
profile was:

```text
QWEN_SD_INT8=1
QWEN_SD_AMX_D=1
QWEN_SD_AMX_BF16=0
QWEN_SD_STREAM_STRIP=1
QWEN_SD_RAG_MIN_PANELS=2
QWEN_SD_POOL=engine
QWEN_BLAS_OWN=1
QWEN_DECODER_BATCH=1
QWEN_STREAM_DECODE_CHUNK=q   # q = 2, 4 or 8 per arm
QWEN_SD_FUSED_RESIDUAL=1
QWEN_SERVER_ASYNC_OUTPUT=0
QWEN_PREFIX_CACHE=1
QWEN_PREFILL_MATMAT=1
QWEN_CP_PREFILL2=1
QWEN_POOL_SPIN=4096
OPENBLAS_THREAD_TIMEOUT=1
```

`QWEN_STREAM_DECODE_CHUNK_BUSY` was absent/default-off. The server's existing startup
ramp (1, 2, 4 before the configured steady-state quantum) was unchanged. No intrusive
profiler or census was enabled, and the receive-fidelity check reported zero
coalesced reads in every arm.

## Results

All values are p50/p95 unless a single value is shown. RTF is dimensionless; times are
milliseconds; `B` is effective batch and `cores` is measured core-equivalent use.

| arm | TTFB | TTFA | STREAM_RTF | stream margin | TOTAL_RTF | req/s | effective B | cores | errors / rejects / timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fused + q2 | 52.9 / 58.7 | 163.9 / 170.5 | 0.866 / 0.914 | 0.134 / 0.086 | 0.927 / 0.982 | 1.57 | 2.39 | 8.8 | 0 / 0 / 0 |
| fused + q4 | 53.6 / 66.9 | 170.9 / 175.5 | 0.827 / 0.868 | 0.173 / 0.132 | 0.903 / 0.942 | 1.64 | 2.37 | 8.8 | 0 / 0 / 0 |
| fused + q8 | 52.9 / 58.0 | 165.2 / 170.1 | 0.773 / 0.822 | 0.227 / 0.178 | 0.846 / 0.886 | 1.74 | 2.36 | 9.0 | 0 / 0 / 0 |

| arm | required prebuffer | safe play start | max gap | stall rate @100 / @250 / @500 / @1000 | stall max p95 | total stall p95 @100 / @250 / @500 / @1000 | coalesced reads |
|---|---:|---:|---:|---|---:|---|---:|
| fused + q2 | 185 / 276 | 345 / 450 | 314 | 83.3% / 0% / 0% / 0% | 133 | 136.7 / 0 / 0 / 0 | 0.0% |
| fused + q4 | 155 / 201 | 329 / 362 | 344 | 33.3% / 0% / 0% / 0% | 129 | 64.4 / 0 / 0 / 0 | 0.0% |
| fused + q8 | 175 / 306 | 335 / 476 | 531 | 58.3% / 25% / 0% / 0% | 142 | 164.9 / 8.1 / 0 / 0 | 0.0% |

The q4 row meets the provisional strong-candidate rule for this screen:
STREAM_RTF p95 `0.868 <= 0.90`, required prebuffer p95 `201 ms <= 300 ms`,
stall@250 `0%`, and no material TTFA regression. Relative to q8, q4 gives up about
`0.036` of STREAM_RTF p95 and `0.10 req/s`, but reduces required prebuffer p95 by
`105 ms`, safe-play-start p95 by `114 ms`, max-gap p95 by `187 ms`, and eliminates
the q8 25% stall rate at a 250 ms buffer. Relative to q2, q4 has slightly higher
TTFA/max-gap but better STREAM_RTF, prebuffer, safe-play-start and throughput.

## Interpretation

1. **F1 outcome — q4 is the current C4 playback/realtime Pareto candidate.** It has a
   useful 13.2% p95 stream margin, no 250 ms stalls in this bank, and TTFA p95 175 ms.
   It should replace q8 as the next C4 reference candidate for a longer qualification,
   not be called production-qualified yet.

2. **q2 is viable but not preferred.** It remains below the mandatory STREAM_RTF p95
   limit, but at 0.914 it misses the preferred 0.90 target and does not improve the
   prebuffer or throughput envelope over q4. This is not a global rejection of q2;
   it is the F1 decision for this exact C4 arm.

3. **q8 remains the throughput control.** It has the best STREAM_RTF and req/s, but its
   playback envelope is worse: max gap 531 ms, prebuffer p95 306 ms, and 25% of
   requests stall with a 250 ms fixed buffer. Aggregate RTF alone would select the
   wrong serving point.

4. **The causal “fused shifted the frontier” claim remains unproven.** F1 establishes
   the joint fused-on q2/q4/q8 frontier and shows that q4 is operationally attractive.
   It does not isolate the effect of fused residual because no exact paired fused-off
   arm was retained, and the earlier non-identical quantum control already showed q2/q4
   below the preferred STREAM p95 target. The safe claim is “fused + q4 is the next
   measured candidate,” not “fusion moved the crossover by a measured amount.”

5. **No AMX share is fabricated.** The clean binary was built with `SIMD=amx` and the
   runtime selected `QWEN_SD_AMX_D=1`; prior qualification proves the Design-D decoder
   path. F1 intentionally omitted census/profiler overhead, so per-arm AMX execution
   remains configuration-backed rather than newly counted in this KPI run.

## Decision

PROMOTE q4 as the next C4 reference candidate for a longer, citation-grade
qualification. KEEP q8 as the throughput/control arm. KEEP q2 as a lower-quantum
fallback candidate, but do not select it as the preferred F1 policy. Do not change
topology, batching, threshold, admission, output, scheduler or kernel behavior based
on this screen.

The result supports a lead-sized-quantum investigation later, with q4 as the measured
floor candidate for this configuration. It does not justify implementing that scheduler
now, nor does it answer C5/cap-3 admission capacity.

## Run hygiene

Two preliminary attempts were excluded from the evidence: q2/q4 trials with a busy
override and mismatched run conditions produced service-cap/RTF failures, and one q8
attempt used an incorrect model path. They were terminated and are not part of the
tables above. The reported arms are the final sequential seed-2027 runs with no
survivor benchmark process, no service-cap event, and zero coalescing.

## Next action

The next recommended falsifier is **F2 — C5 startup decomposition with parent
timestamps**, using q4 as the C4 reference candidate. Do not run F2, cap=3, scheduler,
SL-1, overlap or backend experiments as part of this checkpoint.
