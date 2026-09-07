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

## GCP C3/C4 A/B extension

The implementation was then tested at the first production-relevant development points
using the same clean AMX binary built from runtime commit `04f00d3`. The host remained a
single-socket/NUMA c4-standard-24 with 12 physical cores, SMT off, CPUs `0-11`, 2x6
prefork, engine pool, cap 2, q8, ragged threshold 2, 1.7B INT8 and Design-D. Each arm
used two true simultaneous waves, the same 21-text bank and seed base; the control ran
sequentially with `QWEN_SERVER_ASYNC_OUTPUT=0`, and the treatment changed only that flag
to `1`. No profiler or census was enabled. The binary SHA-256 prefix was
`8358ca9c520e0986`; the transferred archive had no `.git`, so the harness source field
was not authoritative and the explicit runtime commit is the source identity.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p95 ms | safe start p95 ms | max gap p95 ms | stall@250/@500 | coalesced reads | errors/rejects/timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| synchronous control | 3 | 218/435 | .590/.775 | .605/.883 | 321 | 754 | 524 | 33%/0% | 6.1% | 0/0/0 |
| detached writer | 3 | 251/439 | .617/.765 | .648/.818 | 311 | 749 | 527 | 33%/0% | 0.0% | 0/0/0 |
| synchronous control | 4 | 424/503 | .790/.821 | .817/.975 | 367 | 753 | 574 | 25%/0% | 5.1% | 0/0/0 |
| detached writer | 4 | 429/502 | .789/.830 | .836/.985 | 387 | 816 | 572 | 25%/0% | 0.0% | 0/0/0 |

The async writer did not materially change inference or playback KPIs in this small
matched wave: C3 STREAM_RTF p95 moved by `-0.010`, and C4 by `+0.009`; C4 core
equivalent stayed at `7.1` in both arms and effective batch stayed `1.97`. It did remove
already-queued receive coalescence in this run and preserved separate enqueue/write
telemetry. Peak PSS was about `8.7/9.0 GB` at C3/C4 in both arms; no error, reject,
timeout, failed enqueue or writer failure occurred. These values are a Tier-A integration
and resource screen, not a long-concurrency qualification or proof that detached writers
improve throughput.

## Current status / follow-up

PROMOTE as an isolated P3 implementation milestone and retain default-off. The C3/C4
extension passes the integration gate, but does not justify changing the default: the
detached-thread memory/thread cost and behavior under longer concurrency remain unknown.
Before changing the default, run a longer C3/C4 comparison and inspect writer thread count,
RSS/PSS, queue peaks, client-backpressure termination and cadence. Keep MT-4 open for
header/flush timestamp semantics; `[OUT]` timestamps are the current transport owner
telemetry. The lead-gate experiment is a separate rejected serving policy; do not combine
the two features in an attribution run.
