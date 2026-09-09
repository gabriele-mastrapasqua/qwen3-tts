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

## 7. Resource-allocation A/B: 5+3 and 6+2 (2026-09-09 23:33-23:38, same setup as §5)

DL-1 architecture promoted by the owner; the 4+4 allocation was not. Only the two other
splits were run, bounded: fixed text B2/B3/B4, long bank B4, plus the inline long-B4 control.
Gate for the host screen: B4 STREAM p95 ≤ 0.92 on the fixed text AND ≤ 0.90 on the long
bank with stall@250 = 0.

| split (step+dec) | fixed B2 | fixed B3 | fixed B4 | long B4 | stall@250 long B4 | B4 iteration wall mean / p95 | B4 Talker+CP | B4 mailbox wait /frame | dec core-eq (of N) |
|---|---|---|---|---|---|---|---|---|---|
| 4+4 (§5) | 0.774 | 0.871 | **0.997** | not run here | — | 72.3 / 77 | 69.8 | 3.6 ms | 0.82 / 4 |
| 5+3 | 0.761 | 0.894 | 1.113 | 1.015 | 75 % | 80.1 / 136 | 66.9 | 8.0 ms | 0.77 / 3 |
| 6+2 | 0.801 | 1.069 | 1.364 | 1.263 (prebuffer 5.9 s) | 100 % | 97.3 / 215 | 62.9 | 16.6 ms | 0.65 / 2 |
| inline 8T | 0.866 | 1.020 | 1.203 | 1.107 (prebuffer 2.5 s) | 100 % | 92.4 / 192 | 54.0 | — | — |

**Verdict: NO host screen — neither split meets the gate, both are worse than 4+4 at B4.**
What the split says (MEASURED):

* The decoder needs ≥ 4 cores to stay hidden at B4 with q4: on 3 cores the producer starts
  blocking on its slot's mailbox (8 ms/frame, `decode_ms` 12.9 on the loop), on 2 cores it
  blocks 17 ms/frame and the lane is slower than inline. The decoder team is not idle by
  choice at 4+4 (0.82 core-eq): it is idle because units arrive in bursts and are served FIFO.
* Returning threads to the step side buys little: Talker+CP at B4 is 69.8 ms on 4 threads,
  66.9 on 5, 62.9 on 6, 54.0 on 8 inline. The step inflation is roughly half thread count
  and half contention with the decoder team (the inline 8-thread figure is the floor).
* 4+4 stays the allocation of record for this architecture on this host; the remaining
  gap to the gate at B4 is ~0.08 of STREAM p95 on a 2 s clip, i.e. the pipeline latency
  plus the per-slot step work, not the decoder.

No further runs. What changed: this section; PLAN DL-1 line.

## 8. DL-2 — elastic 8 ↔ 4+4 and fast handoff (2026-09-09 23:54, same single-CCX setup)

Built (`QWEN_SD_LANE_ELASTIC=1` with `QWEN_SD_LANE_SPLIT=4`, default-off): the engine pool
keeps all 8 cpus with one worker pinned per cpu (caller on cpu 0); while at least one decoder
unit is queued or running the pool's dispatch width is capped to the 4 STEP cpus
(`qwen_pool_set_width`, honoured by `qwen_parallel` and by `qwen_parallel_team()`, which the
Talker/CP regions re-read every frame), and the width returns to 8 as soon as the decoder
queue drains; both transitions happen under the mailbox mutex. The lane workers park
quickly between units (they share cpus 4-7 with pool workers 3-6). Handoff: one
preallocated job and codes buffer per slot, no heap allocation per unit, the only copy is
the ≤ 2 KB of codes; the wake-up stays a condvar (one per unit, ~30 ms apart). Smoke B1
clean, overruns 0.

| B | inline | static 4+4 | **elastic** | Talker+CP ms inline / static / elastic | decoder in flight (elastic) |
|---|---|---|---|---|---|
| 1 | 0.674 | 0.613 | 0.615 | 44.3 / 47.9 / 47.8 | 14 % |
| 2 | 0.866 | 0.774 | 0.747 | 48.0 / 58.4 / 56.3 | 18 % |
| 3 | 1.020 | 0.871 | 0.859 | 50.1 / 63.5 / 62.6 | 22 % |
| 4 | 1.203 | 0.997 | **0.987** | 54.0 / 69.8 / **69.5** | 31 % |
| long B4 | 1.107 | 0.906 | **0.895** | 54.5 / 71.3 / 70.5 | 50 % |

(STREAM p95; stall@250 = 0 % on every lane row; long-bank static 4+4 B4 measured in this
batch too: 0.906, TOTAL 0.917, prebuffer 466 ms.)

**The static-partition tax is falsified as the cause of the step inflation.** With the
engine on 8 threads for 69-86 % of the loop, Talker+CP at B4 is 69.5 ms against 69.8 ms
static: identical. Thread count was already known to be irrelevant for the weight stream
(bandwidth-bound at 2-4 threads) and the 5+3/6+2 runs had shown 5-6 threads buy 3-7 ms; the
elastic run closes the question. The inflation is concentrated in the windows where the
decoder team is actually running: with the decoder in flight 31 % of the time and a mean
inflation of +15.5 ms per iteration, the step runs ~50 ms slower during those windows
(INFERRED from the shares); at B1 (14 %) the same arithmetic gives ~+25 ms. CP carries most
of it (+10 ms mean at B4, i.e. its L3-resident rows are being evicted by the decoder's
activations while a unit runs), the Talker the rest (+6). This is the L3-pollution term the
contention falsifier (§10 of the fast-screen addendum) under-estimated with the standalone
decoder bench: the real per-item decoder pollutes more.

Gate: fixed B4 0.987 > 0.92 → no host screen; long B4 0.895 meets its half (≤ 0.90,
stall@250 0). Not promoted; both lane modes stay default-off.

What the split now says the lever is: not threads, not the handoff (mailbox wait 0.1-3.4
ms/frame, no allocation), not the decoder's compute, but **the cache footprint of the
decoder unit while it runs next to the CP** — the f32 im2col/ConvT activations of the
per-item path. The next falsifier is a decoder unit with a smaller working set (direct
ConvT / no im2col materialisation, int8 activations, or `res1` on int8 panels that do not
materialise f32), measured by CP ms during overlap, not by STREAM.

## 9. DL-3 falsifiers: what does NOT move the interference tax (2026-09-10 00:16-00:37)

First-class metric added: `[STAGE] overlap=1` marks iterations that start with a decoder
unit in flight; the parser splits every stage by (active, overlap). Baseline (elastic 4+4,
q4, long bank, B4): pure-overlap iterations CP **35.7 ms** vs 23.5 without the decoder (+12,
i.e. the CP at its pure-DRAM rate, 1.8 GB / 55 GB/s = 33 ms: its L3-resident half is gone);
Talker 36-40 vs 30.6 inline (+6-10). The tax per overlapped iteration is ~+20 ms; at B4 the
decoder is in flight 48 % of the loop (units of 60-65 ms per 4 frames on 4 threads, four
of them per 290 ms quantum).

| arm (all elastic 4+4, one CCX, 1.7B) | CP in overlap (long B4) | STREAM p95 long B4 / fixed B4 | decoder unit (4 frames) | note |
|---|---|---|---|---|
| base q4 | 35.7 | 0.895 / 0.987 | 60-65 ms | reference |
| `QWEN_SD_DIRECT_CONVT/DWCONV/INPUT=1` | 36.3 | 0.889 / 0.981 | same | the existing direct paths: no effect |
| sub-quantum decode, 1-frame sub-calls (`QWEN_SD_LANE_SUBQ=1`) | 38.2 | 1.475 / 1.517 | 31 ms per frame | live set /4 makes it WORSE: 4x the calls saturate the lane (73 % in flight, mailbox wait 10 ms/frame) |
| sub-quantum, 2-frame sub-calls | 36.8 | 1.132 / 1.196 | 46 ms per 2 frames | same direction |
| q8 units (`QWEN_STREAM_DECODE_CHUNK=8`) | 34.2 | **0.864** / 1.025 | 113 ms per 8 frames | passes 0.90 on the long bank but prebuffer 806 ms, stall@250 100 %: the cadence law, not admissible |
| NTA prefetch of conv weights in the VNNI tile (`QWEN_SD_NTA=1`) | 37.2 | 0.893 / — | same | no effect |
| lane workers hot during a unit (park only between units) | 35.7 | 0.895 / 0.978 | same | no effect on the unit time |
| decoder panels sized by the lane team (bug fix, kept) | 35.7 | 0.902 / 0.991 | same | no effect on the unit time |

Reading (MEASURED, HIGH): the interference tax is not the decoder's activation live set,
not its weight stream's cacheability, not the handoff, not park latency, and not the unit
size. Whenever the decoder team runs beside the step team on the same CCX, the CP loses its
L3 residency and the Talker's per-slot working set suffers, by a roughly constant ~20 ms per
iteration. The only quantity that scales the mean tax is the FRACTION of iterations
overlapped, which is the decoder's time on its four cores: 15-16 ms per frame there against
7.6 on eight threads inline. Halving the decoder unit time halves the overlap share and the
mean tax (≈ −8 ms per iteration at B4 → Talker+CP ≈ 62 ms → B4 ≈ 0.90).

**DL-3 verdict: cache-friendliness tricks are falsified; the lever is decoder kernel
efficiency on the lane (res1 = 52 % of the unit, at ~5 % of VNNI peak; the panel build is
~30 % of the conv time; the 2x4 tile re-reads each weight row 32 times per panel).**
That is the next implementation (DL-4): a res1 path with fewer weight re-reads and no
separate f32 panel materialisation, measured by decoder unit time on 4 threads and by
`overlap` share, then STREAM. Not started here.

Code kept from this cycle (all default-off or bug fixes): `overlap` field in `[STAGE]`,
`QWEN_SD_LANE_SUBQ`, `QWEN_SD_NTA`, hot lane workers during a unit, panel sizing by the lane
team. Raw runs: `~/bench/lane/dl3*`, `~/bench/lane/c-dl2b-*` on the host.
