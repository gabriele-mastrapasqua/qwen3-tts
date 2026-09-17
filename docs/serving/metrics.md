# Watching a running server — the metrics endpoint

[← Serving index](README.md)

`--metrics-port` publishes what the server already knows, in Prometheus/OpenMetrics text, on a
port of its own. It works for **both backends**: the CPU server with or without `--prefork`, and
a GPU server, which is always single-process because a CUDA or Metal context does not survive
`fork()`.

```bash
./qwen_tts -d qwen3-tts-1.7b --serve 8080 --prefork 12 --batch-size 2 \
           --metrics-port 9109
curl -s http://127.0.0.1:9109/metrics
```

Off by default. Bound to `127.0.0.1` unless you pass `--metrics-bind`, and rate limited to
**5 scrapes per second** unless you pass `--metrics-max-rate` (see below).

## What this is, and what it is not

It is a **rendering of state the server was already maintaining for its own scheduling**, done
when somebody asks for it. Nothing is counted for the sake of the page, nothing runs between
scrapes, and no code sits on any path a request travels. The server pushes nothing and keeps no
telemetry history: collection, retention, rates and alerting belong to whatever is polling.

It is **not the acceptance instrument.** A `histogram_quantile` in Grafana is an estimate from
buckets; the p95 that qualifies a box comes from raw samples in
[`cpu-operations.md`](cpu-operations.md)'s harnesses. Use the scrape to see drift, a starving
worker, or rejection pressure. Never quote it as a qualification number — same discipline as the
`[MEASURED]` / `[PREDICTED]` labels in `make doctor`.

And it is **not the whole envelope.** Half of what acceptance is judged on is a client-side
measurement by construction:

| KPI | on this page? | why |
|---|---|---|
| in-flight, queued, dispatched, completed, rejects | **yes** | server-side state, already maintained |
| audio seconds produced, TTFA, TTFB | **yes** | measured in the worker, from instants the request already carries |
| chunks arriving behind realtime | **yes, as a named proxy** | the server sees its own writes, not the listener's buffer |
| `required_prebuffer`, `safe_play_start`, `stall_rate@N` | **no** | a stall is an event in a *player*; the server does not hold the playback clock |

A server-side "stall rate" would be an invented number. It is not exported, deliberately. The
nearest honest thing is the behind-realtime gap count, and it is named so it cannot be mistaken
for one.

## Reading it with standard tools

The page is Prometheus/OpenMetrics text, so it is read directly by Prometheus, Grafana Alloy,
VictoriaMetrics and the Datadog OpenMetrics check — and by the **OpenTelemetry Collector**
through its `prometheus` receiver, which converts to OTLP for anything downstream. That is how
this reaches OTel consumers without a protobuf client entering a server that has no
dependencies.

```yaml
# prometheus.yml
scrape_configs:
  - job_name: qwen-tts
    static_configs: [{ targets: ["your-host:9109"] }]
```

Note on the GenAI semantic conventions: as of 2026-09 every `gen_ai.*` metric is still
stability *Development*, the namespace was moved out of the main semconv repository in v1.42.0,
and it defines **no audio or text-to-speech operation at all**. There is nothing there to be
compatible with yet, so the series below use a `qwen_tts_` namespace and plain underscores.

## The series

Everything is labelled `worker` and **nothing is summed for you**. A reader can always add
per-worker series up; it cannot take an average apart again. "One worker wedged while the others
absorb the load" — `inflight` pinned at `slots` with a flat `completed_total` — is exactly the
failure a summed view hides, and the one the soak campaigns kept finding.

| series | type | meaning |
|---|---|---|
| `qwen_tts_build_info{git_rev,source_fp,simd,mode}` | gauge, always 1 | which binary is running, and whether it is `prefork` or `single` |
| `qwen_tts_workers` | gauge | processes serving requests |
| `qwen_tts_worker_up{worker}` | gauge | that worker process is alive |
| `qwen_tts_worker_inflight{worker}` | gauge | requests in flight on it |
| `qwen_tts_worker_slots{worker}` | gauge | concurrent requests it will accept |
| `qwen_tts_worker_dispatched_total{worker}` | counter | requests handed to it |
| `qwen_tts_worker_completed_total{worker}` | counter | requests it finished |
| `qwen_tts_worker_waiting{worker}` | gauge | *single process only* — queued, not yet started |
| `qwen_tts_worker_cpus{worker}` | gauge | *`--prefork-elastic` only* — CPUs currently assigned |
| `qwen_tts_elastic_replans_total` | counter | *`--prefork-elastic` only* — re-slicings |
| `qwen_tts_rejected_total{reason}` | counter | refusals, by reason |
| `qwen_tts_metrics_throttled_total` | counter | scrapes this endpoint refused with `429` |
| `qwen_tts_worker_audio_seconds_total{worker}` | counter | **seconds of audio produced** — `rate()` is realtime streams sustained |
| `qwen_tts_worker_requests_finished_total{worker}` | counter | requests finished, where the timings are taken |
| `qwen_tts_worker_ttfa_seconds_sum{worker}` / `_count` | counter | time to first audio; `rate(sum)/rate(count)` is the windowed mean |
| `qwen_tts_worker_ttfa_over_250ms_total{worker}` / `_over_500ms_` / `_over_1s_` | counter | **the shape of the tail** — exact counts past each line, per worker |
| `qwen_tts_worker_terminated_ok_total{worker}` | counter | ran to completion and was delivered |
| `qwen_tts_worker_terminated_client_gone_total{worker}` | counter | **client disconnected mid-stream** |
| `qwen_tts_worker_terminated_timeout_total{worker}` | counter | stopped by the per-request budget |
| `qwen_tts_worker_terminated_rejected_total{worker}` | counter | refused after admission |
| `qwen_tts_worker_ttfb_seconds_sum{worker}` / `_count` | counter | time to first byte |
| `qwen_tts_worker_queue_seconds_sum{worker}` / `_count` | counter | **admission wait** — enqueue to the scheduler taking it |
| `qwen_tts_worker_stream_gaps_total{worker}` | counter | chunk-to-chunk gaps measured |
| `qwen_tts_worker_stream_gap_behind_realtime_total{worker}` | counter | gaps where wall time exceeded the audio delivered |
| `qwen_tts_worker_stream_gap_over_1s_total{worker}` | counter | gaps over 1 s outright |

### Why audio seconds and not requests per second

A request is not a unit of work here: one is two seconds of speech, the next is sixty.
`rate(qwen_tts_worker_audio_seconds_total[1m])` is how many **realtime listeners** the box is
carrying, and it is the number that means something for a TTS server. Requests per second is
kept because a worker whose request rate goes flat is wedged, which is a different question.

### Why three thresholds instead of a percentile

A percentile is estimated from the samples in its tail, and there are `N x (1-q)` of those. At
C16 this server does roughly 1.8 requests/second, so a two-minute run is ~215 requests and a
**p99 is the second-worst request** — the count above it is `Binomial(N, 0.01)`, mean 2.15,
standard deviation 1.5. It can legitimately move 50% between identical runs. That is not a
number a comparison can rest on, and `tests/load_test.py` now prints a bootstrap confidence
interval and the supporting sample count beside every percentile, marking any that rests on
fewer than ten samples.

Exact threshold counts do not have this problem: with 215 requests, *"zero crossed 1 s"* is
precise, while *"p99 = 253 ms"* is not. Three lines — 250 ms, 500 ms, 1 s — give the shape of
the tail per worker at any sample size, with no histogram buckets. When a percentile really is
the question, the arithmetic is simple: ~9 minutes per arm for a thousand requests, ~18 for two
thousand.

### Why counting how requests end matters

A listener closing the tab mid-stream is the most ordinary event a TTS server sees, and before
these counters it was invisible on every series here: the queue stays shallow, nothing is
refused, throughput simply sags. `terminated_client_gone` makes a step change in abandonment
something you can see and alert on, and separates "people are leaving" from "we are failing".

### Why admission wait is separate from TTFA

First audio can be late for two unrelated reasons: the request waited behind other work, or it
was slow once it started. Those want opposite responses — more capacity versus a faster engine —
and without the split they are the same graph. Subtracting the queue mean from the TTFA mean
separates them, and both instants were already on the request.

### Why "behind realtime" and not a millisecond threshold

The first version of this counted chunk gaps longer than 250 ms, and on a healthy stream it
fired on almost every chunk — because a chunk carrying 500 ms of audio that arrives 400 ms after
the previous one is **filling** the listener's buffer, not draining it. The gap only matters
against the audio delivered in it. Measured on an M1 with the same text and chunking: bf16
(RTF ≈ 1.3) reported 13 behind-realtime gaps out of 13, `--int8` (RTF ≈ 0.85) reported **0 out
of 12**. The metric discriminates the thing it claims to.
| `qwen_tts_timed_out_total{worker}` | counter | *single process only* — over the request budget |

Useful queries:

```promql
rate(qwen_tts_worker_completed_total[5m])                      # throughput, per worker
qwen_tts_worker_inflight / qwen_tts_worker_slots               # saturation, per worker
sum(rate(qwen_tts_rejected_total[5m])) by (reason)             # pressure, and of what kind
max(qwen_tts_worker_inflight) - min(qwen_tts_worker_inflight)  # imbalance
```

## Rate limiting — because the page is rendered in the dispatch loop

The same decision that makes this endpoint free at a sane scrape interval is what makes a
runaway client dangerous: in `--prefork` the page is rendered **inside the parent's dispatch
loop**. A `watch -n 0.01 curl` or a scraper misconfigured to 10 ms would buy itself a hundred
renders a second out of the budget that hands requests to workers.

So the endpoint serves at most `--metrics-max-rate` scrapes per second (**default 5**) and
refuses the rest with `429 Too Many Requests` and a `Retry-After`. `--metrics-max-rate 0`
disables the limit.

- **A token bucket, not a minimum interval**, for the same reason DynamoDB uses one: a fixed
  floor punishes the legitimate case — Prometheus and somebody's `curl` landing in the same
  millisecond — while doing nothing about a sustained flood. The bucket holds `2 × rate` tokens,
  so short bursts pass and the average is capped.
- **A refusal is cheaper than an answer**, or the limit funds the attack: the `429` path does
  not render the page.
- **Refusals do not consume tokens.** A client hammering at 100/s gets the configured rate
  served and the rest refused, rather than locking out everyone including itself.
- **The refusals are on the page**: `qwen_tts_metrics_throttled_total`. Nonzero means something
  is polling too fast — not that the server is unhealthy.

Default sizing: a 5 s `scrape_interval` is 0.2 scrapes/s, and even three independent scrapers
plus `metrics_watch.py` at 1 s stays under 1.5/s. The default of 5/s is generous for anything
real and still caps an accident at a cost the parent cannot feel.

**Honest scope: this is QoS against accident, not DDoS protection.** A hostile flood is a
firewall's problem, and the port is loopback-bound by default precisely so that it is not
reachable to flood in the first place.

## Two honest limits, stated on the page itself

**In `--prefork`, the rejects are the parent's only.** `reason="all_workers_full"` and
`reason="fd_dispatch_failed"` are decisions the parent made. A rejection inside a worker's own
queue happens in that worker and is invisible from here — the page says so in its `HELP` text
rather than letting the number read as a total.

**A plain single server publishes identity only.** Without `--batch-size 2` or more and without
`--prefork`, there is no scheduler and the request counters are never incremented by that path.
Rather than publish a page of zeros that look like measurements, those series are **absent** and
the startup log says why. Run with `--batch-size` or `--prefork` to get counters.

## Checking that it adds up

```bash
tools/metrics_watch.py --url http://127.0.0.1:9109/metrics --duration 120 --interval 10
```

Samples the page while load runs elsewhere and checks the three properties a scraper cannot
survive being wrong: **monotonic** (no counter went backwards — Prometheus reads a decrease as a
restart and silently drops the interval), **conserved** (`completed <= dispatched`, always), and
**balanced** (every live worker received something).

`configs/observability/alerts.yml` ships Prometheus rules for the conditions worth waking
somebody for. Every one of them fires on something a **listener** would notice — first audio
past the budget, chunks arriving with the buffer draining, refusals, a lost worker, a step
change in abandonment — and every threshold is an **exact count, never an estimated quantile**,
for the reason above.

`tests/serve_metrics.sh` is the regression gate: it proves the flag is refused outside server
mode and on the service port, that a plain server omits counters it does not maintain, that a
batched server's counters move with traffic, that nothing listens by default, that fast polling
is refused with `429` while a served scrape stays a complete page and `--metrics-max-rate 0`
turns the limit off, and — on Linux — that the prefork parent emits one distinct series set per
worker and every worker moves.

## Seeing it in a dashboard

`configs/observability/` is a working example — a Prometheus scrape config and a Grafana
dashboard built on the per-worker series — with `tools/observability_up.sh` to fetch and start
both, provisioned, bound to loopback:

```bash
tools/observability_up.sh                                  # on the box
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 user@box # from your laptop
```

Then `http://127.0.0.1:3000` → **Qwen3-TTS serving**. Read
[`configs/observability/README.md`](../../configs/observability/README.md) for what each panel
is built to expose, and why nothing there is a qualification number.

## Cost

**Measured on a 32-core Neoverse-V2 (GCP Axion), 2026-09-17.** Three arms of a closed-loop C16
soak, 120 s each, on the frozen `axion-c4a-highcpu32-0p6b-all-on` profile (0.6B int8, 4 workers
× 8 threads, batch 8). The arms differ **only** by `--metrics-port`; the middle one also had
`tools/metrics_watch.py` scraping every 5 s, so the cost includes real scrapes. The OFF arm was
run first and last, because the interesting comparison is not on-versus-off but
**on-versus-the-box's-own-drift**.

| | off (first) | on | off (last) |
|---|---|---|---|
| completed | 217 | 217 | 213 |
| errors | 0 | 0 | 0 |
| TTFB p50 / p95 (ms) | 23.0 / 75.1 | 18.4 / 73.1 | 21.3 / 75.2 |
| TTFA p50 / p95 (ms) | 116.8 / 200.5 | 106.9 / 198.8 | 124.5 / 190.1 |
| TTFA p99 (ms) | 242.0 | 212.6 | 212.4 |
| RTF p50 / p95 | 0.76 / 0.82 | 0.75 / 0.80 | 0.77 / 0.85 |

The two OFF arms — identical binary, identical flags, identical seed — already differ by 6.6% on
TTFA p50 and **12.2% on p99**. Every on-versus-off delta is inside that, and several ON numbers
are *better* than both OFF arms, which cannot be a real effect. The honest conclusion is that
any cost is **below the noise floor of a 120 s window on this box**, not that the endpoint is
free in some stronger sense. Tested at a 5 s scrape interval; 1 s was not measured.

Nothing is added to any path a request travels, and no worker changes at all. In `--prefork` the
page is rendered from inside the parent's dispatch loop, on the one thread that owns that state,
so there is no shared memory, no snapshot and no atomic. Reads and writes to the scrape socket
are non-blocking and one-shot: a scrape that would block is **dropped** rather than retried,
because losing one sample is cheaper than delaying the server that produced it.

One bounded exception, and it was found by measuring rather than by reading. `accept()` can
return before the request bytes land; answering and closing right then leaves the request
arriving at a closed socket, the kernel replies `RST`, and the scraper discards a response that
was already written. Under load that appeared as roughly **one empty scrape in three hundred**.
The handler therefore waits up to **2 ms**, once, for the request — generous on loopback, and
invisible beside the dispatch loop's own 1000 ms `poll()`.
