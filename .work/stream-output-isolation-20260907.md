# OUT-1/OUT-2 streaming output isolation — 2026-09-07

## Decision

Implemented and kept independently switchable behind `QWEN_SERVER_ASYNC_OUTPUT=1`.
The default remains the existing synchronous writer until a longer serving qualification
establishes the thread/memory cost at the target concurrency.

The feature gives each streaming request an owned bounded PCM queue and a detached socket
writer. The inference callback only converts/enqueues PCM; it never writes to the socket
or waits for a reader. Queue overflow, allocation failure, disconnect, send error or send
timeout marks the stream failed, wakes the producer, and closes the stream. PCM is never
silently dropped. The old path remains the control when the flag is unset or writer setup
fails.

The same isolation is wired through both paths:

* continuous ragged-batch jobs (`sink_next_job` / `sink_on_chunk`);
* the single-job streaming path used for clone/design requests.

`qwen_compose_chunk_cb` now propagates a non-zero callback result so a failed output can
stop markup composition rather than continuing to synthesize discarded audio.

## Local gates

Implementation commit: `78d0426` (`feat: isolate streaming output from inference`).

* `make clean && make blas`: PASS (native local build).
* `./qwen_tts --self-test`: PASS, 0 failures.
* `python3 tools/check_flag_registry.py`: PASS, 197/197.
* `python3 tools/check_plan.py`: PASS.
* `python3 tools/check_repo_integrity.py`: PASS with the pre-existing missing
  `docs/emotion-seeds.md` warning.
* `git diff --check`: PASS.

## GCP Tier-A path A/B

Host: `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA, SMT `off`, online CPUs
`0-11`, topology `2x6`, 1.7B INT8, engine pool, batch cap 2, Design-D decoder settings,
ragged threshold 2, chunk 8. Runs used `tests/serve_parallel_wave.py`, true simultaneous
wave, one wave at C1/C2, no profiler/census. The transferred source was local commit
`78d0426`; the tar snapshot has no `.git`, so the harness source field is `unknown` and
the binary identity is the authoritative `sha256` prefix `a9e0435fa25edbdb`.

| arm | C | TTFA p50/p95 (ms) | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 (ms) | safe start p50/p95 (ms) | max gap p95 (ms) | @500 stall | errors/rejects | coalesced reads |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synchronous control | 1 | 98/98 | .519/.519 | .527/.527 | 13/13 | 111/111 | 329 | 0% | 0/0 | 6.7% |
| synchronous control | 2 | 105/115 | .546/.579 | .555/.593 | 23/28 | 128/143 | 373 | 0% | 0/0 | 7.4% |
| detached writer | 1 | 99/99 | .516/.516 | .524/.524 | 13/13 | 112/112 | 329 | 0% | 0/0 | 0.0% |
| detached writer | 2 | 105/115 | .547/.574 | .556/.588 | 26/28 | 131/143 | 380 | 0% | 0/0 | 0.0% |

This is a path/integration gate, not a Tier-B qualification. The KPI differences are
within this one-wave sample's noise; the intended measurable change is TTFB/transport
ownership, not an inference-speed claim. The detached writer logs showed all normal streams
with `failed=0`, `failed_enqueues=0`, peak queued PCM `30720` bytes, and distinct enqueue
and write timestamps.

## Audio parity

The same C1 bank/seed was run sequentially with output off and on using `--save-audio`.
All four matching WAVs were byte-identical (`cmp` and SHA-256): conversational, Italian,
long and medium; sizes ranged from 199724 to 956204 bytes. This checks conversion,
chunk framing and queue ordering on the continuous batch path.

## Slow/stopped reader test

Dedicated temporary raw-socket test on port 9430 used the same AMX binary and topology.
The first client read only the HTTP header, advertised a 1 KiB receive buffer and then
read nothing for 12 seconds. The second client was a normal full reader. The feature used
`QWEN_STREAM_OUTPUT_MAX_BYTES=65536` and `QWEN_STREAM_OUTPUT_SEND_TIMEOUT_MS=1000` only
to force the bounded failure path.

* slow client: HTTP 200 header, detached writer logged `failed=1` after 28 enqueued
  chunks / 378240 samples; the send timeout closed that stream; `failed_enqueues=0`
  correctly distinguishes a socket failure from queue overflow;
* normal client: HTTP 200, 2304820 bytes, completed in 39.209 s while the slow client
  remained unread; its writer logged `failed=0` after 79 chunks / 1152000 samples;
* no qwen process survived the explicit server shutdown.

This proves the bounded writer/cancellation path and that a stalled socket is not executed
on the inference callback. It is not a claim that the per-stream detached-thread design is
the final high-concurrency transport architecture.

## Current status / follow-up

PROMOTE as an isolated P3 implementation milestone and retain default-off. Before changing
the default, run a longer C3/C4 comparison with many concurrent streams and inspect writer
thread count, RSS/PSS, queue peaks, client-backpressure termination and cadence. Keep MT-4
open for header/flush timestamp semantics; `[OUT]` timestamps are the current transport
owner telemetry. The next independent runtime target is the minimal per-stream lead/credit
skeleton; do not combine it with this patch.
