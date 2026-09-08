# F2 — C5 startup decomposition (2026-09-08)

Task · F2 C5 startup decomposition; diagnostic instrumentation plus two short server arms.

Question · Is the multi-second C5 startup tail predominantly hidden before prefork acceptance/admission, or is there a comparable post-admission bottleneck?

Known facts · The source was clean at `fd9c5d4` on `feature/x86-amx-vnni-oss`. The reference was a GCP `c4-standard-24` with Xeon Platinum 8581C, one socket/NUMA, 12 physical CPUs online (`0-11`), SMT off, 2x6 prefork, cap 2, Design-D INT8 decoder, ragged threshold 2, engine-owned decoder pool, and fused residual enabled. The clean AMX build reported `SIMD=amx`, source fingerprint `fd9c5d4:clean`, and binary SHA-256 prefix `fcefe0a7c7082caf`. The startup log reported 24 persistent INT8 decoder B packs (19.3 MB) and the resolved AMX decoder configuration.

Unknowns · Userspace cannot timestamp a TCP connection before `accept()` without invasive packet tracing; `client_start -> parent_accept` is therefore a derived backlog-wait estimate. The three-wave workload co-launched five requests, rather than admitting one request after four streams were already established, so established-stream causality is limited. KPI runs intentionally had no profiler/census; this note does not claim a TDPBSSD tile count. In synchronous mode `write_attempt/write_complete` identify the response-header write at admission, not the later PCM socket write; `first_pcm` is the inference-side callback before that PCM write.

Files/functions inspected · `qwen_tts_server.c`: `setup_listen_socket`, `qwen_tts_serve_prefork`, `srv_send_fd`, `srv_recv_fd`, `cq_pop`, `reader_main`, `jq_push`, `sink_next_job`, `sink_on_chunk`, `qwen_life_emit`; `qwen_tts.c`: continuous serving loop, `TTFA2`, `F2STAGE`, decoder dispatch and `T2_FIRST_AUDIO`; `tests/serve_parallel_wave.py`: true-wave client timestamps, request metrics and playback simulation; `docs/BENCHMARKING.md`: diagnostic server-argument and timestamp-header rules.

Evidence · The exact clean AMX build ran three true-simultaneous C5 waves with the 1.7B INT8 model and the same q4 reference profile. Arm A used the normal full-cap parent behavior. Arm B added only `--max-queue 0`. Both had `QWEN_TTFA_TRACE=1` for the timeline and no profiler/census. Client receive coalescing was 0% in both arms. Raw artifacts remain outside the tracked repository; this file records their public-safe summary.

Conclusion · The multi-second tail is overwhelmingly a parent/listener admission problem under the tested cap-2/full-cap behavior. Three Arm-A requests waited 3.63–4.89 s before `accept()`, then reached first PCM 87.5–132.9 ms after engine admission. The derived pre-admission share of client-start-to-first-PCM was 97.25–98.25% for those three requests; even the pre-accept share was 94.57–97.47%. Arm B accepted full arrivals and returned three immediate parent-side 503 responses; its 12 accepted requests had no multi-second accept wait. A secondary engine-queue/prefill term remains measurable for normally accepted requests, but it is hundreds of milliseconds, not the C5 multi-second tail.

Next action · F-cap3 is justified as the next capacity falsifier, but was not run here. No admission policy or capacity behavior was changed by F2.

## 1. Ground truth and setup

The tested runtime profile was:

```text
QWEN_SD_INT8=1 QWEN_SD_AMX_D=1 QWEN_SD_AMX_BF16=0
QWEN_SD_STREAM_STRIP=1 QWEN_SD_RAG_MIN_PANELS=2
QWEN_SD_POOL=engine QWEN_BLAS_OWN=1 QWEN_DECODER_BATCH=1
QWEN_STREAM_DECODE_CHUNK=4 QWEN_SD_FUSED_RESIDUAL=1
QWEN_SERVER_ASYNC_OUTPUT=0 QWEN_PREFIX_CACHE=1 QWEN_PREFILL_MATMAT=1
QWEN_CP_PREFILL2=1 QWEN_POOL_SPIN=4096 OPENBLAS_THREAD_TIMEOUT=1
QWEN_TTFA_TRACE=1 QWEN_LIFE_TRACE=1
```

The harness used `TRUE_SIMULTANEOUS_WAVE`, 3 waves, 5 requests per wave, the realistic mixed text bank, and `--topo 2x6 --batch-cap 2 --precision int8`. The binary was built clean from `fd9c5d4`; the harness's `dirty=yes` field is a known artifact of running from a source snapshot without a `.git` directory, not a dirty source tree. The binary's embedded source fingerprint and the independent remote snapshot fingerprint were clean.

## 2. Code-path audit

| boundary | current behavior at the frozen source |
|---|---|
| listen backlog | `listen(fd, 16)` in `setup_listen_socket`; this is the kernel backlog configured by the process, not a measured number of pending requests. |
| parent listener poll | `qwen_tts_serve_prefork` adds the listener to `poll()` only when at least one worker has `active[w] < cap`, or when `g_cfg_max_queue == 0` enables `reject_full_at_parent`. |
| slot accounting | `cap = max_batch` (2 here); parent `active[w]` counts whole connections dispatched to each worker and is decremented only by the worker completion notification. |
| worker choice | after `accept()`, the parent selects the least-loaded worker with `active[w] < cap`; `free_slots_at_accept` is recomputed after accept for the trace. `parent_slot` is a diagnostic timestamp immediately after accept, not a separate kernel admission event. |
| fd handoff | parent sends the accepted fd plus optional F2 metadata over the worker Unix socketpair; the child records `child_receive`. The handoff itself is sub-millisecond in the tested rows. |
| child reader | `reader_main` reads and parses the complete HTTP request, then pushes a `batch_job_t` into the engine job queue. `recv`, `parsed`, and `enqueued` are recorded in this path. |
| child queue | with the ordinary unset queue setting, `jq.cap` becomes one grace slot. `jq_push` limits `running + queued` against `slots + cap`; `sink_next_job` applies the queue timeout only after a request has reached this child queue. |
| engine admission | `sink_next_job` pops a job, checks the queue deadline, increments running/admitted, records `t_admit`, and sends the synchronous chunked response header before entering the engine loop. |
| generation/decode | the continuous loop records `step1`, decoder entry, and first audio; decoder output reaches `sink_on_chunk`, where `t_first`/`first_pcm` is set before synchronous PCM transmission. |

Therefore the synthesis claim is correct with one qualification: the fifth connection is not necessarily delayed, but it can remain in the kernel listen backlog whenever all four whole-connection parent slots are occupied and the default full-cap policy excludes the listener from `poll()`. The child queue timeout cannot bound this interval because the parent has not accepted or dispatched that connection. With `--max-queue 0`, the listener remains polled and the same full condition produces an immediate parent-side 503.

## 3. Instrumentation

Commit `637cae7` added the default-off F2 handoff and request timeline. Commit `fd9c5d4` added the separate `[F2STAGE]` first-Talker-phase marker. The existing `QWEN_TTFA_TRACE` switch controls the diagnostic path; when disabled, the client sends no F2 header, no handoff metadata is sent over the prefork channel, and no F2 request timestamps are read. The timestamps use `CLOCK_MONOTONIC` in the server and Python `time.monotonic()` in the client.

The server emits one structured `[PATH]` record per completed request and `[F2REJECT]` for parent-side full-cap rejects. The client header aligns `client_start` with the server monotonic domain. `[TTFA2]` supplies the engine phase timestamps already present in the runtime. The new `[F2STAGE]` marker is useful but must be interpreted with the actual loop order: the current continuous loop performs the first decoder work before the first later Talker phase marker. Consequently `step1 -> first_talker_phase` is measurable, but `first_talker -> first_decode` is not a valid positive interval for this path; it is explicitly not used as a causal duration below. The `[TTFA2]` line can print `talker1=0` because it is emitted at first audio before the later Talker-phase marker is reached.

## 4. Startup timeline

Values are `median/p95` in milliseconds, reconstructed from the accepted main-bank seeds. Arm A has a second view containing its three parent-backlog-delayed requests. `NA` means that the population has no such request, not zero latency.

| stage | Arm A, all 15 accepted | Arm A, delayed 3 | Arm B, 12 accepted |
|---|---:|---:|---:|
| client start -> parent accept | 0.3 / 4666.7 | 4666.7 / 4885.2 | 0.3 / 0.4 |
| parent accept -> parent dispatch | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| parent dispatch -> child receive | 0.0 / 0.2 | 0.0 / 0.0 | 0.0 / 0.2 |
| child receive -> request read complete | 0.2 / 0.5 | 0.1 / 0.2 | 0.2 / 0.4 |
| request read -> parsed | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| parsed -> enqueued | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| enqueued -> engine admitted | 39.2 / 185.4 | 39.2 / 120.6 | 33.5 / 187.2 |
| admitted -> prefill start | 0.0 / 0.0 | 0.0 / 0.3 | 0.0 / 0.1 |
| prefill start -> prefill end | 113.5 / 365.0 | 51.5 / 52.9 | 113.6 / 370.7 |
| prefill end -> first generation loop (`step1`) | 1.2 / 317.7 | 0.3 / 0.3 | 28.1 / 322.9 |
| first generation loop -> first decoder | 15.0 / 17.1 | 13.6 / 15.0 | 16.3 / 16.8 |
| first generation loop -> first Talker-phase marker | 51.0 / 55.2 | 35.7 / 81.9 | 52.9 / 56.9 |
| first Talker-phase marker -> first decoder | **not ordered in this loop; UNKNOWN** | **not ordered** | **not ordered** |
| first decoder -> first PCM inference callback | 35.2 / 38.7 | 22.0 / 66.9 | 37.1 / 40.1 |
| engine admitted -> first PCM inference callback | 278.4 / 481.7 | 87.7 / 132.9 | 368.5 / 564.2 |

The three Arm-A delayed requests were seeds 5046, 5051 and 5056 in the client bank. Their individual client-start-to-first-PCM durations were approximately 3837.8, 4835.6 and 5012.2 ms. Their parent-accept waits were approximately 3629.4, 4666.7 and 4885.2 ms; engine-admit-to-first-PCM was approximately 87.7, 132.9 and 87.5 ms.

## 5. Population classification

| population | count | classification |
|---|---:|---|
| Arm A promptly accepted at parent | 12 | accepted and dispatched promptly; some then waited in the child/engine queue (individual `enqueued -> admitted` reached about 426.7 ms). |
| Arm A accepted only after full-cap interval | 3 | delayed before parent `accept()` while the listener was absent from `poll()`; parent dispatch after acceptance was immediate. |
| Arm A rejected | 0 | none. |
| Arm B accepted | 12 | parent accepted immediately; remaining post-accept work followed the normal child/engine path. |
| Arm B parent fail-fast | 3 | `[F2REJECT] reason=all_workers_full`, `free_slots_before=0`, `free_slots_at_accept=0`, `cap=2`; the parent returned 503 without reading the request body, so the reject records have no request seed. |
| admitted but timed out/error | 0 | no execution timeout or server computation error in either arm. |

The three Arm-B harness `errors` are the three expected HTTP 503 rejects, not failed accepted inferences. They are excluded from accepted-request latency and playback percentiles.

## 6. Client and playback KPIs

All latency/playback values below are for accepted requests; `errors/rejects` is shown separately. Receive coalescing was 0% in both arms.

| metric | Arm A full-cap, 15 accepted | Arm B `--max-queue 0`, 12 accepted + 3 rejects |
|---|---:|---:|
| TTFB p50/p95 | 77.5 / 4703.6 ms | 67.6 / 188.6 ms |
| TTFA p50/p95 | 424.6 / 4838.8 ms | 539.7 / 593.8 ms |
| STREAM_RTF p50/p95 | 0.818 / 0.852 | 0.782 / 0.824 |
| TOTAL_RTF p50/p95 | 0.865 / 3.470 | 0.828 / 0.876 |
| required prebuffer p50/p95 | 200.2 / 352.7 ms | 140.7 / 155.1 ms |
| safe play start p50/p95 | 694.0 / 4965.8 ms | 642.9 / 735.6 ms |
| max gap p50/p95 | 315.3 / 374.7 ms | 298.1 / 398.5 ms |
| stall rate @100 / @250 / @500 / @1000 | 60% / 0% / 0% / 0% | 33.3% / 0% / 0% / 0% |
| total stall @100 p95; @250/@500/@1000 | 164.7 ms; 0 / 0 / 0 | 10.5 ms; 0 / 0 / 0 |
| request rate | 0.288 req/s | 0.240 req/s |
| effective B | 2.364 | 2.195 |
| measured cores | 7.33 | 7.91 |
| errors / rejects / timeouts | 0 / 0 / 0 | 3 HTTP 503 / 3 / 0 |
| receive coalescing | 0% | 0% |

Arm B's accepted-set throughput must not be interpreted as extra capacity: it processes fewer requests because three arrivals were rejected. Its value is causal: full-cap arrivals become explicit fail-fast responses instead of hidden multi-second waits.

## 7. Effect on co-launched peers

This was a true simultaneous five-request wave, not a dedicated “four established streams, then admit a fifth” experiment. Therefore it cannot prove the full established-stream admission-interference question.

Within the tested waves, the first four Arm-A peers had no fixed-buffer stall at 500 ms; Arm B's accepted peers also had no stall at 500 ms. Their observed max-gap envelopes were of the same order (about 404 ms worst among Arm-A first-four peers versus about 399 ms p95 for Arm-B accepted peers). Arm A did show safe-play-start values up to about 961 ms among those first-four peers, but the wave does not isolate whether that came from the fifth connection, request ordering, or ordinary worker/prefill interference. The defensible result is: no direct @500 continuity hole was observed in these accepted peers, but established-stream protection remains unmeasured as a clean causal scenario.

## 8. Measured versus derived quantities

Measured directly from the server/client traces:

- parent accept, dispatch, child receive, request-read/parse/enqueue, engine admission, prefill, generation-loop, decoder entry and inference-side first PCM timestamps;
- `[F2REJECT]` full-cap records in Arm B;
- TTFB, TTFA, STREAM_RTF, TOTAL_RTF and playback simulation metrics;
- topology, CPU masks, SMT state, build/source fingerprint, resolved runtime flags and zero coalescing.

Derived from aligned monotonic timestamps:

- `client_start -> parent_accept` as a pre-accept/backlog-wait estimate;
- all stage durations and the pre-admission/post-admission fractions;
- the interpretation that a full-cap listener was not polled, based on the current parent control flow.

Not measured in this diagnostic:

- kernel-level connection arrival before the userspace client timestamp/`accept()`;
- a direct TDPBSSD/TDPBF16PS census or AMX MAC share (profiler/census were excluded from KPI runs);
- a clean experiment in which one new request is admitted after four already-streaming requests;
- separate completion timing for every synchronous PCM socket write.

## 9. Causal verdict

**Result A confirmed for the multi-second C5 tail.** Under the normal full-cap arm, all three pathological requests were accepted only after a 3.6–4.9 s pre-accept wait. More than 97% of each client-start-to-first-PCM interval occurred before engine admission, while post-admission first PCM took only 87.5–132.9 ms. The parent dispatch and child handoff were effectively immediate after acceptance. This is direct evidence for hidden fixed-slot/listener-backlog wait, not decoder arithmetic explaining the seconds-long latency.

There is a real secondary term after acceptance: ordinary accepted Arm-A requests had `enqueued -> admitted` median/p95 about 33.4/185.4 ms, prefill median/p95 about 118.6/365.0 ms, and engine-admit-to-first-PCM median/p95 about 358.0/481.7 ms. This term can matter to C5's normal accepted tail and future admission policy, but it does not replace the primary explanation for the multi-second outliers.

Arm B provides the matching falsifier: with only `--max-queue 0` changed, three full-cap arrivals were accepted by the parent and rejected immediately, while the 12 accepted requests had no multi-second pre-accept tail and TTFA p95 593.8 ms. This does not increase capacity and does not qualify C5; it demonstrates hidden wait versus explicit overload signaling.

## 10. Next action

F-cap3 is the next justified capacity falsifier because F2 establishes that the current cap-2 failure is primarily an admission/backlog boundary. It must separately measure whether a larger cap improves useful accepted capacity without turning the secondary engine/prefill term or established-stream continuity into the new failure. F-cap3 was intentionally **not** executed in this task.
