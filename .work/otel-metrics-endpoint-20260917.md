# Telemetry for the streaming server — analysis before implementation

Addendum to PLAN task **OTEL-1..OTEL-5**. Written 2026-09-17, from the tree and from the
state of the OpenTelemetry GenAI conventions on that date.

The ask: expose metrics for the streaming server in an OpenTelemetry-compatible way, so that
standard readers can consume them. This document is the analysis that must happen **first**;
no endpoint is written until OTEL-1 and OTEL-2 are decided.

## 0. Summary of the finding

Three things came out of the survey, and the second one is the one that shapes the design.

1. **The transport is not the hard part.** A `/metrics` page in Prometheus/OpenMetrics text
   format is ~40 lines of `snprintf` and is readable by Prometheus, Grafana Alloy, the
   OpenTelemetry Collector (`prometheus` receiver, converted to OTLP downstream),
   VictoriaMetrics and the Datadog OpenMetrics check. Producing OTLP directly would drag a
   protobuf/gRPC client into a dependency-free C server for no reader we do not already reach.
2. **The GenAI semantic conventions cannot express our acceptance envelope, and they do not
   cover audio at all.** They are token-centric. Half of what we qualify on is a *client-side*
   playback measurement that a server cannot honestly claim to know. What the server exports
   and what stays harness-only is the real decision here — not the wire format.
3. **The server is prefork, and its counters are process-local.** `/v1/health` already returns
   one worker's view of a multi-worker server. That is a pre-existing defect that any metrics
   endpoint would inherit and multiply.

## 1. State of the OpenTelemetry GenAI conventions (as of 2026-09-17)

- Every `gen_ai.*` attribute, span, metric and event still carries stability **Development**.
  None is Stable.
- On **2026-06-12**, with semantic-conventions **v1.42.0**, the GenAI conventions were
  deprecated out of the main semconv repository into a dedicated repository,
  `open-telemetry/semantic-conventions-genai`, which as of mid-July 2026 had no tagged release.
- Server-side metrics defined there:

  | metric | instrument | unit |
  |---|---|---|
  | `gen_ai.server.request.duration` | histogram | `s` |
  | `gen_ai.server.time_to_first_token` | histogram | `s` |
  | `gen_ai.server.time_per_output_token` | histogram | `s` |

- Client-side, for streaming callers: `gen_ai.client.operation.duration`,
  `gen_ai.client.operation.time_to_first_chunk`,
  `gen_ai.client.operation.time_per_output_chunk`, `gen_ai.client.token.usage`.
- Explicit bucket boundaries for `time_to_first_token`:
  `[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]`.
  Note the `1.0` boundary — our `safe_play_start` hard line is exactly 1 s, so this set is
  usable for us rather than merely tolerable.
- `gen_ai.operation.name` enumerates `chat`, `text_completion`, `embeddings`,
  `generate_content`, `invoke_agent`, `execute_tool`, `retrieval`, the memory operations —
  **no `text_to_speech`, no audio operation of any kind.**

Consequence: adopting `gen_ai.server.*` names buys compatibility with dashboards that do not
exist yet, for a convention that can rename its fields without a deprecation window, in a
domain it does not model. It is a *candidate alias set*, not the primary namespace.

## 2. The de-facto precedent: vLLM

vLLM is what an operator's dashboard is already shaped around, so it is the compatibility
target that actually pays.

- Prefix `vllm:`, every series labelled `model_name`.
- Gauges: `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`.
- Counters: `prompt_tokens_total`, `generation_tokens_total`,
  `request_success_total{finished_reason=stop|length|abort}`.
- Histograms: `time_to_first_token_seconds`, `inter_token_latency_seconds`,
  `e2e_request_latency_seconds`, `request_queue_time_seconds`, `request_prefill_time_seconds`,
  `request_decode_time_seconds`.
- Their own docs concede the colon in `vllm:` is contrary to Prometheus convention (`:` is
  reserved for recording rules). **Do not copy that.** Use `qwen_tts_` with `_` separators.
- They document a deprecation policy — notice in the HELP string, release-note entry, a CLI
  escape hatch that restores a removed metric for one cycle. Worth copying verbatim as policy.
- They also document that built-in process metrics break under multiprocess
  (`--api-server-count > 1`). Which is our problem too, below.

**We already have vLLM's gauge set.** `GET /v1/health` returns `num_requests_running`,
`num_requests_waiting`, `queue_max`, `queue_timeout_ms`, `max_request_ms`, `max_text_chars`,
`admitted`, `done`, `rejected_queue_full`, `rejected_queue_timeout`, `timed_out`
(`qwen_tts_server.c:862`). It is the right data in the wrong encoding, behind no scraper.

## 3. What the server already measures and throws away

`batch_job_t` (`qwen_tts_server.c:1584`) carries, per request:

- `t_recv`, `t_parsed`, `t_admit`, `t_first`, `t_write_attempt`, `t_write_complete`
- `first_audio_ready_us`, `audio_ready_samples` (atomics, written by the audio sink)
- the prefork handoff instants: `t_parent_accept`, `t_parent_slot`, `t_parent_dispatch`,
  `t_child_receive`, plus `parent_worker`, `free_slots_at_accept`, `parent_cap`
- outcome flags: `client_gone`, `cancelled`, `timed_out`

So queue time, admission time, TTFB, TTFA, handoff cost, audio seconds produced and the
terminal reason are **all already computed** and discarded when the job is freed. The work is
aggregation and exposition, not instrumentation. This matters for the cost argument: we are not
adding clock reads to the hot path, we are adding one atomic increment per bucket at job
completion.

## 4. The blocker to decide first: prefork counters are per-process

`static server_state_t g_srv;` (`qwen_tts_server.c:597`) is a plain process-local static. After
`fork()` each worker owns a private copy. `/v1/health` is answered inside the worker that
received the connection handoff (`reader_main`, `qwen_tts_server.c:1857`, dispatch at `:1881`).

**Therefore, today, a health probe against a 12-worker prefork server already returns one
worker's counters — roughly 1/12 of the traffic, and a different twelfth each time.** The
*limits* it reports (`queue_max`, `max_text_chars`, …) are correct and identical across
workers; the *counters* are not the server's. `docs/serving/api.md:256` says health "reports the
same numbers live, which is how a client discovers the limits it is subject to" — true for the
limits, misleading for `admitted`/`done`/`rejected_*`. Verify empirically, then either fix or
document.

Two ways out, and they are not equivalent:

- **(a) Shared segment, parent aggregates.** The pattern already exists in this file:
  `qwen_admission_health_t` is an `mmap(MAP_SHARED|MAP_ANONYMOUS)` array with one slot per
  worker and atomic fields, allocated at `qwen_tts_server.c:2804-2822` and bound per worker by
  `qwen_admission_health_bind()`. Extending that segment with a counter/bucket block per worker
  costs one mmap and no new IPC. Required for gauges (`running`, `waiting`) — a sum of stale
  per-worker gauges is not a gauge.
- **(b) Per-worker labelled series.** Expose `worker="3"` on every series and let the reader
  `sum()`. Correct and free for counters and histogram buckets (a sum of bucket counts is the
  global bucket count). Wrong-ish for gauges unless all workers are scraped in the same instant.

Likely answer is both: shared segment for the few gauges, `worker` label for counters and
histograms so a slow or wedged worker stays *visible* instead of being averaged away. That
visibility is worth the cardinality: `workers × ~10 series` is small, and "one worker is
starving" is exactly the failure the soak work kept finding.

## 5. The honest-metric problem — what a server can and cannot claim

Our acceptance envelope is: TTFB, TTFA, `required_prebuffer`, **`safe_play_start` < 1 s (hard
line)**, `max_gap`, `stall_rate@{100,250,500,1000}`, `STREAM_RTF`, rejects, errors.

Split by who can observe it:

| KPI | server can export honestly? | why |
|---|---|---|
| TTFB | **yes** | `t_first - t_recv`, both server-side |
| TTFA | **yes** | `first_audio_ready_us - t_recv` |
| queue / admission time | **yes** | `t_admit - t_parsed` |
| request duration | **yes** | `t_write_complete - t_recv` |
| audio seconds per wall second | **yes** | `audio_ready_samples / 24000` over elapsed |
| `STREAM_RTF` | **yes, with care** | wall/audio per request; it is a **mean rate**, and a mean rate is the metric our own rule says fails *last* |
| **cadence debt / write-gap** | **yes, as a named proxy** | gap between successive chunk writes vs audio duration in the chunk — server-observable |
| `required_prebuffer` | **no** | defined against a player's consumption clock |
| `safe_play_start` | **no** | derived from a playback simulation over the whole arrival trace |
| `stall_rate@N` | **no** | a stall is an event in a *player*, not in the server |
| `max_gap` | partial | server sees its own write gaps, not the listener's |

This is the core of the analysis and the thing to get right before writing code. Exporting a
server-side `stall_rate` would be inventing a number: the server does not know the client's
playback clock. What it *can* export is **cadence debt** — audio produced minus audio that
should have been produced by now, per stream — which is the server-side shadow of a stall and
must be named so that nobody quotes it as the harness's `stall_rate`.

Corollary, and it belongs in the docs next to the metric: **a Grafana `histogram_quantile` p95
is not a qualification number.** Bucketed quantiles are estimates; our qualification p95 comes
from raw samples in `tools/`-side harnesses. The scrape is the **operations** instrument
(is it drifting, is a worker starving, page someone); `load_test.py` / `serve_parallel_wave.py`
/ the soak remain the **acceptance** instrument. Same discipline as the `doctor` provenance
labels: the instrument determines what you are entitled to claim.

## 6. Sketch of the namespace (to be ratified, not yet implemented)

```
# gauges (shared segment)
qwen_tts_requests_running{worker}
qwen_tts_requests_waiting{worker}
qwen_tts_workers_total
qwen_tts_slots_total{worker}

# counters
qwen_tts_requests_total{worker,route,outcome}         # outcome: ok|rejected_full|rejected_stale|timeout|client_gone
qwen_tts_audio_seconds_total{worker}
qwen_tts_frames_total{worker}

# histograms (semconv TTFT buckets)
qwen_tts_time_to_first_byte_seconds{worker,route}
qwen_tts_time_to_first_audio_seconds{worker,route}
qwen_tts_queue_time_seconds{worker}
qwen_tts_request_duration_seconds{worker,route}
qwen_tts_stream_rtf                                    # unitless, own buckets
qwen_tts_write_gap_seconds{worker}                     # the cadence-debt proxy, NOT stall_rate

# build/config identity, value 1, everything in labels
qwen_tts_build_info{version,isa,model_size,profile}
```

**Label cardinality is a hard constraint.** `voice` must NOT be a label: the preset set is
bounded at 9 but the clone path accepts arbitrary names, so a label on voice is an unbounded
series generator driven by user input. Same for `language` and any text-derived field. Allowed
label set: `worker`, `route`, `outcome`, plus the static identity labels on `build_info`.

## 7. Endpoint placement and exposure

- Path `/metrics`, **not** `/v1/metrics`: scrapers default to `/metrics`, and `/v1/metrics` is
  what OTLP-over-HTTP uses for its own push endpoint — colliding with it would be a lasting
  confusion.
- **Opt-in** (`--metrics` flag or env), off by default. A metrics page is an information
  disclosure surface: it publishes concurrency, capacity, model size and build identity to
  anyone who can reach the port.
- In prefork mode the page has to be served by the parent (it is the only process with the
  whole picture) or by any worker reading the shared segment. Decide with OTEL-2.
- It must answer without taking any synth lock, and must not be subject to `--batch-size`
  admission — a metrics scrape that queues behind TTS work is a metrics endpoint that goes
  blind exactly when you need it.

## 8. Cost gate

Instrumentation is not free until measured. The bar already used for the cost-map work applies:
added overhead **< 0.2 %** of wall, verified with `make cost-map` and a C12 wave A/B on the
frozen Turin profile, before the endpoint is documented as production-safe.

## 9. Open questions (this is the analysis, OTEL-1)

1. Prometheus-text only, or also an OTLP push path later? (Leaning: text only; the Collector
   bridges it.)
2. Shared segment vs per-worker labels vs both (§4).
3. Do we emit `gen_ai.server.*` aliases for GenAI-aware readers, given Development stability and
   the repo move — or wait for a tagged release of `semantic-conventions-genai`?
4. Do we propose `text_to_speech` / an audio operation upstream? Nothing in the convention
   covers it, and we have an unusually well-specified envelope to argue from.
5. Is `/v1/health` per-worker today (§4)? If yes: fix, document, or both — and does anything in
   `tests/` depend on the current behaviour?
6. Exact bucket sets for `stream_rtf` and `write_gap` (the semconv TTFT set covers the latencies).
7. Is `/metrics` inside or outside the privacy boundary — does `build_info` leak anything
   `private/guard/privacy_check.sh` would refuse in a tracked artifact?

## Sources consulted

- `open-telemetry/semantic-conventions-genai`, `docs/gen-ai/gen-ai-metrics.md`
- OpenTelemetry semantic-conventions v1.42.0 deprecation of `gen_ai.*` (2026-06-12)
- vLLM `docs/design/metrics.md` and the v1 metrics design page
- The tree: `qwen_tts_server.c` (`g_srv` :597, `handle_health` :862, `batch_job_t` :1584,
  `reader_main` :1857, admission-health mmap :2804), `qwen_tts.h:689`, `docs/serving/api.md`
