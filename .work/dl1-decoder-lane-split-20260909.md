# DL-1 — 4+4 intra-CCX decoder lane: implementation and A/B (2026-09-09)

Status · IMPLEMENTED default-off; A/B DONE. **Verdict: the pre-registered GO gate is NOT met (B3 STREAM p95 0.871 vs ≤ 0.85, B4 0.997 vs ≤ 0.92) and the FAIL conditions are NOT triggered (11-17 % better than inline at B2-B4, overlap real, mailbox wait 1.6-3.6 ms/frame, lifecycle clean). Not promoted; kept default-off. The measured split names the next lever: the STEP side on 4 threads, not the decoder (§6).**

Task · DL-1 (PLAN P4). Question · Does taking the per-item speech decoder off the frame
loop's critical path, onto a private team on the last 4 cores of an 8-core worker, move a
1.7B lane from B2 to B3/B4 at the realtime gates, without lifecycle regressions?

Pre-registered predictions (from `.work/turin-c8a-32c-fast-screen-20260908.md` §10):
B3 ≈ 64 ms per frame (≈ 0.80), B4 ≈ 72 ms (≈ 0.90). Predictions, not pass criteria.

Gate · GO: B3 STREAM p95 ≤ 0.85, B4 ≤ 0.92, stall@250 = 0, no lifecycle/correctness issue.
Strong GO: B4 ≤ 0.90 and no material TTFA/prebuffer regression. FAIL: < 10 % better than
inline at B3/B4, or Talker/CP inflation erases the overlap, or the mailbox recreates
equivalent blocking, or lifecycle becomes unsafe.

## 1. Known facts the design rests on (MEASURED, one CCX of c8a.8xlarge, 1.7B int8)

| fact | value | source |
|---|---|---|
| inline lane law (fixed text, STAGE) | `T(B) = 40 + 13.5·B` ms; STREAM p95 .674/.861/.991/1.172 at B1..B4 | fast-screen §8 |
| per-slot cost | decoder 9.7 ms per decoded slot-frame (72 %), Talker +1.6, CP +1.8, serial 0.05 | same |
| Talker+CP vs threads | 26.4 + 20.8 ms on 2 threads = 26.2 + 18.1 on 8: the weight stream saturates the CCX at 2-4 threads | T1-b |
| decoder vs threads (ms/frame) | 33 (1 core) · 18.5 (2) · 12.9 (4) · 7.6-9.7 (8) | T1-b / §8 |
| two weight streams on one CCX | halve each other (1.28 vs 0.97) | T1-a |
| decoder-only load on 6 cores beside a 2-core step lane | step lane +12 % (CP 21 → 24, residency kept) | L3 falsifier §10 |
| per-slot region sections on 2 threads | +17 ms per slot (vs +6.4 on 4, +3.4 on 8) → the step side needs 4 threads | §10 |

## 2. What was built (code, default-off)

* `qwen_tts_thread.{c,h}`: `qwen_lane_split_prepare()` reads `QWEN_SD_LANE_SPLIT=N`,
  takes the calling worker's affinity mask (whatever cpuset it inherited), sorts it, gives
  the FIRST `n−N` cpus to the engine (STEP) and the LAST `N` to the decoder, and confines
  the calling thread to STEP before the engine pool is created (pthreads inherit the mask).
  `qwen_lane_team_start()` creates `N−1` workers pinned to the decoder cpus (`sd-lane-*`)
  with their own generation/spin/park scheme and their own submit lock; the decoder thread
  (`dec-thr`) joins the lane with `qwen_lane_thread_join()`. **The single choke point**:
  `qwen_parallel()` redirects every dispatch made from a lane thread to the lane team, so
  conv panels, snake rows, SGEMM slices, im2col and the decoder's bf16 matmat never reach
  the engine pool or `P.submit_mtx`; nested dispatches inside a lane task run inline.
* `qwen_tts_kernels.c` / `qwen_tts_sd_gemm.c`: `sd_pool_run` and the SGEMM slicer size
  their work by the lane team when called from it.
* `qwen_tts.c`: the existing `QWEN_DECODER_THREAD` consumer (cloned decoder context,
  per-slot busy count, finalize/`on_done` from the decoder thread) is reused with the lane
  flag: `dec_wait_idle()` is the bounded mailbox — before enqueuing a slot's next quantum,
  or its final unit, or freeing its state on cancel, the loop waits for THAT slot's unit
  only; other slots' steps never wait for a decode. Batching in the consumer is off (per
  item on VNNI anyway). `[STAGE]` gains `dec_wait_ms`; `[serve-profile]` prints
  `[lane-cpu]` (core-equivalents per team from `/proc/self/task`) and `[lane-contract]`
  (mailbox overruns, must be 0). The team is stopped when the loop exits (no orphan).
* `qwen_tts_server.c` (prefork child) and `main.c` (`--cpu-mask` single worker) call the
  split before `qwen_set_threads`; `--dispatch-map` reports `decoder.lane` with the
  resolved masks; `QWEN_SD_LANE_SPLIT` is in the declared flag registry.
* Profiles: `configs/perf/vnni-bf16-product.json` (native bf16 prefill for CPUs with
  `avx512_bf16`; `parity.backends.prefill = bf16-native`) and `QWEN_POOL_SPIN=65536` in both
  VNNI product lanes with the measured why. Schema, README and tests updated.

Not done, on purpose: deadline ordering (FIFO per submission), res1 work, any admission,
topology or batching change. Bound: at most one decoder unit in flight per slot; produced
but not yet enqueued frames ≤ one quantum (the loop blocks at the next quantum boundary).

## 3. Smoke (B1, one CCX, q4, split 4)

`[lane] split prepared: step cpus 0-3 (4, engine pool) · decoder cpus 4-7 (4, private team)`;
STREAM p95 0.613 (inline q4 ≈ 0.70), iteration wall mean 48.2 ms, p95 53 ms (inline: 52 mean,
81-88 p95 — the decoder-call spikes are gone), `decode_ms` 0.02 (enqueue only), Talker 28.5,
CP 19.5 (the predicted +12 % contention on 4 step threads), mailbox wait 0, overruns 0.

## 4. Correctness gates (all PASS; `~/bench/lane_ab.log`, `~/bench/lane/reorder-*`)

| gate | result |
|---|---|
| clean build (M1 compile check, c8a `make blas`), `--self-test` | PASS (0 failed); `--dispatch-map` row `decoder.lane` OFF by default, ON with the resolved masks |
| waveform parity inline vs lane, same text/seed, temperature 0 | C1 and C4: identical lengths (48000 samples), correlation 0.999961, max diff 259 LSB (FP summation order: 4-thread vs 8-thread SGEMM/conv slices, the benign −90 dB class); on the 5-text bank at C4 per-request correlation 0.99992-0.99997, lengths identical |
| no cross-slot PCM reorder | 5-text bank at C4: each lane WAV matches its own inline counterpart at ≥ 0.9999 and every other text at |corr| ≤ 0.027; four distinct md5 per arm |
| cancellation / disconnect | two clients cut at 1.5 s and 1.0 s into a long stream (curl rc 124), then one full request (1,036,800 B = 21.6 s) and two concurrent full requests (both 1,036,800 B) on the same server: all served |
| final drain | last chunk delivered through the mailbox on every request; `[serve-profile]` accounted all frames |
| no orphan decoder thread/team | thread census 45 before and after the lifecycle run (`dec-thr`, `sd-lane-0..2`, `qwen-pool-0..2` present), SIGTERM → "server exited cleanly", no leftover process |
| mailbox contract | `[lane-contract] mailbox overruns 0` in every run; max in flight per slot 1 |
| resolved masks recorded | `[lane] split prepared: step cpus 0-3 (4, engine pool) · decoder cpus 4-7 (4, private team)` in every server log; the wave's topology JSON records the process mask |

## 5. A/B — one worker `1x8@0-7`, 1.7B int8, one fixed short text (2.0 s), q4, SL-1, profile aws-c8a-16c-vnni-ttfa (+ `QWEN_DECODER_BATCH=0`, chunk 4), one wave

| B | arm | TTFA p95 | STREAM p50/p95 | TOTAL p50/p95 | prebuffer p95 | safe-start p95 | max gap p95 | stall @100/@250/@500 | iteration wall mean / p95 | Talker | CP | decoder on loop | mailbox wait /frame | core-eq step / dec | csw/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | inline | 169 | .853/.866 | .904/.904 | 183 | 352 | 275 | 100/0/0 % | 67.1 / 116 | 27.3 | 20.7 | 18.8 | — | 7.3 (8T) | 4.1k |
| 2 | lane | 198 | .758/**.774** | .826/.826 | 183 | 349 | 279 | **0**/0/0 % | 58.7 / **64** | 32.4 | 26.0 | 0.02 | 1.6 ms | 2.04 / 0.54 | 1.9k |
| 3 | inline | 237 | 1.010/1.020 | 1.079/1.079 | 295 | 532 | 312 | 100/0/0 % | 78.8 / 149 | 27.9 | 22.1 | 28.5 | — | 7.2 | 4.8k |
| 3 | lane | 286 | .855/**.871** | .947/.947 | 314 | 535 | 323 | 100/0/0 % | 64.3 / **72** | 35.1 | 28.4 | 0.5 | 2.4 ms | 2.09 / 0.67 | 2.5k |
| 4 | inline | 311 | 1.193/1.203 | 1.282/1.282 | 542 | 853 | 378 | 100/**100**/0 % | 92.4 / 192 | 30.6 | 23.4 | 38.1 | — | 7.1 | 5.4k |
| 4 | lane | 375 | .980/**.997** | 1.097/1.097 | 465 | 744 | 372 | 100/**0**/0 % | 72.3 / **77** | 36.6 | 33.2 | 2.1 | 3.6 ms | 2.44 / 0.82 | 2.8k |
| 5 | lane | 457 | 1.096/1.128 | 1.249/1.250 | 574 | 906 | 392 | 100/0/0 % | 80.8 / 127 | 38.1 | 34.5 | 8.0 | 6.0 ms | 2.41 / 0.90 | 3.0k |

Decoder service time on the lane team (4 threads, `[SDPHASE]`): 1-frame call 10.5 ms, 2-frame
22 ms, 4-frame 29 ms (inline on 8 threads: 6.4 / 15.3 / 13.7). Effective lead: one unit in
flight per slot, at most one quantum accumulated behind it (the mailbox wait is where the
producer met that bound: 1.6 → 6.0 ms/frame from B2 to B5).

Reading against the pre-registration:

* **Predictions matched on the iteration wall**: B3 64.3 ms (predicted 64), B4 72.3 ms
  (predicted 72). The decoder left the loop (0.02-2 ms of `decode_ms`), the decoder-call
  spikes vanished (wall p95 192 → 77 ms at B4) and stall@250 at B4 went from 100 % to 0 %.
* **The STREAM gate is missed anyway**: B3 0.871 (gate ≤ 0.85), B4 0.997 (gate ≤ 0.92). Two
  reasons, both measured: (1) the STEP side on 4 threads inflated — Talker+CP 50.1 → 63.5 ms
  at B3 and 54.0 → 69.8 at B4 (+27-29 %), i.e. the per-slot region sections (norms, RoPE,
  attention, SwiGLU, quant, argmax: one slot per thread) cost ~8 ms per slot on 4 threads
  against 3.4 on 8; this ate 40 % of the removed decoder time; (2) on a 2 s clip the
  pipeline's fixed latency (a 4-thread ramp decode of 10-29 ms at the start, the last unit at
  the end) adds ~0.05 to STREAM_RTF and 30-64 ms to TTFA; the mean iteration wall (0.80 /
  0.90 of the frame budget) is what a long request would see.
* **FAIL conditions**: not triggered — 11 % (B2), 15 % (B3), 17 % (B4) better STREAM p95
  than inline; Talker/CP inflation reduced but did not erase the overlap (net −14.5 ms at B3,
  −20 ms at B4 per iteration); the mailbox blocked the producer 1.6-3.6 ms/frame, not the
  38 ms the inline decoder cost; no lifecycle issue.
* Decoder team utilization 0.54-0.82 core-equivalents of 4 at B2-B4: the decoder lane is
  idle two thirds of the time; the step lane is the wall now.

## 6. Decision

Binary against the gate: **NOT GO** (and not FAIL). Per the task's rule the 4x8 host screen
is not run. The measured split says the next lever is not res1/VNNI decoder work: the
decoder is off the critical path and under-used; what stands between the lane and B3/B4 at
the gate is the per-slot region work on a 4-thread step team (+8 ms per slot). The one-variable
follow-ups the data supports, in order, none of them run here: (a) `QWEN_SD_LANE_SPLIT=3`
and `=2` (5+3 / 6+2): the decoder at 3 cores ≈ 15 ms/frame still serves B4 in 240 of 320 ms,
and each thread returned to the step side cuts its per-slot cost; (b) the same A/B on the
long bank, where the fixed pipeline latency amortizes and STREAM ≈ wall/80; (c) only then
the res1 kernel. The `vnni-bf16-product` profile and the spin promotion are independent of
this verdict and stand.

What changed: code (`qwen_tts_thread.{c,h}`, `qwen_tts_kernels.c`, `qwen_tts_sd_gemm.c`,
`qwen_tts.c`, `qwen_tts_server.c`, `main.c`, `qwen_tts_dispatch.c`), profiles
(`vnni-product.json` spin, new `vnni-bf16-product.json`, schema, README, tests), PLAN DL-1
(A/B done, not promoted). Raw runs: `~/bench/lane/`, `~/bench/lane_ab.log` on the host.
