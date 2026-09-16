# AMX C4 steady-state attribution and chunk sweep — 2026-09-06

## Decision

The decode-chunk knob changes the short synchronized wave, but it does not yet
qualify the mixed-bank C4 server. The best mixed-bank screen was chunk `32`
with `STREAM_RTF p50/p95 = 0.9099/0.9855`; this is below one but has no useful
margin and was only a 90-second screen. Chunk `24` was worse at
`0.9192/1.0736`. No candidate is promoted and no C5/C6 run is authorized.

The next change must address the measured serving dataflow/panel utilization,
not another isolated tile-kernel tuning loop.

## Identity and controls

- host: GCP `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA;
  CPUs `0-11` online, SMT off;
- topology: two prefork workers × six physical cores;
- model: Qwen3-TTS 1.7B, mixed `tests/load_texts_en.txt` bank, English speaker
  `ryan`, stratified schedule seed `42`;
- AMX binary: source `a4ddf7b:clean`, SHA-256
  `84035115d87550c0761990bac327831d983bf5362aa5b2fca1a03bb960ddba11`;
- Design D INT8 and decoder batching enabled; BF16 decoder disabled;
- `QWEN_SD_POOL=engine` resolved and reported as `engine`;
- canonical valid SOAK control remains commit `3f7e0df`, chunk `8`,
  `--batch-size 2`, five minutes after a 30-second warm-up.

The local SOAK runner now records per-response `underrun_s`, `stall_max_s`,
`prebuffer_s`, `gap_ratio_max` and `chunks`. These are explicitly a
zero-buffer playback diagnostic derived from client chunk timestamps; they are
not server queue counters and are not silently converted into a qualification
gate. The runner change is commit `db8fba6`.

## Existing C4 control

The valid five-minute run at commit `3f7e0df` completed 136 requests, with 120
post-warm-up KPI samples and 16 probes, and zero errors, rejects or timeouts.
At chunk `8`, `STREAM_RTF p50/p95 = 0.9419/1.0350` and TTFA p50/p95 was
`215.2/496.4 ms`. Per-window stream p95 was `0.996, 0.997, 1.096, 1.013,
1.075`. C4 remained unqualified.

The manifest for this control uses server `--batch-size 2`; an old profile
file's batch-size `8` is not the identity of this qualification and must not be
used for comparison.

## Bounded attribution

The non-intrusive short-wave screen at chunk `8` measured aggregate effective
batch about `2.38` (about `1.07` and `1.31` in the two workers), roughly 8.3
core equivalents, and no errors/rejects. A diagnostic ragged-timer run showed
the following source behavior:

- decoder panels claim work over the ragged `N`/panel axis;
- calls with fewer than eight panels execute the worker body serially;
- calls with eight or more panels submit the existing engine-owned decoder pool;
- large `M=768` calls often have only one or two `N` panels, while `M=96/192`
  calls have many panels and are the calls that naturally enter the pool;
- all observed decoder panels used the Design D INT8 AMX path with no fallback.

Representative ragged shapes from the diagnostic run were:

| M | K | Kp | panel cap | typical total N | panels | execution |
|---:|---:|---:|---:|---:|---:|---|
| 96 | 672 | 768 | 128 | 7680 | 60 | engine pool |
| 192 | 1344 | 1536 | 128 | 2560 | 20 | engine pool |
| 384 | 2688 | 2816 | 128 | 640/1280 | 5/10 | serial or pool by call |
| 768 | 5376 | 5376 | 64/128 | 128/256 | 1/2 | serial |

The timers are diagnostic sums across calls and are not reported as wall-time
fractions. They establish the geometry and scheduler policy, not a causal
percentage. The useful conclusion is that server concurrency currently
parallelizes independent panels within each decoder item; it does not create a
larger cross-request logical matrix workset. The observed system batch is still
only about 2.3, and per-worker batches are close to one.

## Short synchronized chunk screen

Three waves, C4, short class, 2x6, no profiler, batch-cap `16` per worker in
the wave harness. This is a screening result, not a mixed-bank SOAK.

| chunk | TTFA p50/p95 ms | STREAM p50/p95 | TOTAL p50/p95 | req/s | B | cores | zero-buffer prebuffer p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8  | 172/181 | 0.874/0.919 | 0.928/0.988 | 1.57 | 2.38 | 8.3 | 0.450 s |
| 12 | 170/182 | 0.863/0.905 | 0.936/0.968 | 1.60 | 2.38 | 8.6 | 0.757 s |
| 16 | 173/187 | 0.853/0.939 | 0.921/1.014 | 1.63 | 2.40 | 8.6 | 0.785 s |
| 24 | 173/179 | 0.828/0.875 | 0.881/0.949 | 1.60 | 2.32 | 8.4 | 0.927 s |
| 32 | 174/176 | 0.839/0.880 | 0.895/0.954 | 1.60 | 2.33 | 8.3 | 1.222 s |

The short wave has only 12 requests per arm and every request has a simulated
zero-buffer gap. It is useful for ranking the knob, not for claiming gapless
production streaming.

## Mixed-bank candidate screens

These repeat the two leading short-wave candidates with the valid SOAK
`--batch-size 2`, full mixed bank, C4, 2x6, 90 seconds, 15-second warm-up,
three 30-second windows, and the same AMX/pool flags. Both had zero errors,
queue rejects and request timeouts.

| chunk | completed/KPI/probes | TTFA p50/p95 ms | STREAM p50/p95 | zero-buffer underrun p50/p95 | window STREAM p95 |
|---:|---:|---:|---:|---:|---|
| 24 | 39/35/4 | 211.3/520.5 | 0.9192/1.0736 | 1.671/1.913 s | 0.960, 1.066, 1.085 |
| 32 | 40/36/4 | 197.9/534.7 | 0.9099/0.9855 | 2.151/2.549 s | 0.931, 1.088, 0.970 |

The per-window class mix is not large enough for a five-minute drift claim, so
these are candidate screens only. They are sufficient to reject the claim that
the chunk knob alone has produced a safe C4 margin. The zero-buffer diagnostic
also shows that a low pooled STREAM_RTF number is not equivalent to gapless
playback without a client buffer policy.

## Configuration and process corrections

`QWEN_SD_POOL` now accepts explicit `engine` and `private` values, preserves
the legacy aliases actually accepted by the old parser, and fails fast on an
unknown explicit value. Startup/effective-config output prints requested and
resolved values. The implementation is in `abd7902`, with the portable remote
test correction in `a4ddf7b`.

## Next bounded step

Keep chunk `32` as a diagnostic candidate, not a promoted default. Measure one
non-intrusive mixed-bank attribution arm with the same batch-size and workload,
then choose one bounded runtime change based on evidence among:

1. useful cross-request aggregation of ready decoder work sharing the same
   persistent B packs;
2. a different ragged panel decomposition for low-panel/high-M calls; or
3. a pool scheduling/batch-cap change that raises useful per-worker work
   without introducing an intentional wait.

Do not run C5/C6 or reopen BF16 until C4 has a stable mixed-bank margin.
