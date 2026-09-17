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

Off by default. Bound to `127.0.0.1` unless you pass `--metrics-bind`.

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
| `required_prebuffer`, `safe_play_start`, `stall_rate@N` | **no** | a stall is an event in a *player*; the server does not hold the playback clock |

A server-side "stall rate" would be an invented number. It is not exported, deliberately.

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
| `qwen_tts_timed_out_total{worker}` | counter | *single process only* — over the request budget |

Useful queries:

```promql
rate(qwen_tts_worker_completed_total[5m])                      # throughput, per worker
qwen_tts_worker_inflight / qwen_tts_worker_slots               # saturation, per worker
sum(rate(qwen_tts_rejected_total[5m])) by (reason)             # pressure, and of what kind
max(qwen_tts_worker_inflight) - min(qwen_tts_worker_inflight)  # imbalance
```

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

`tests/serve_metrics.sh` is the regression gate: it proves the flag is refused outside server
mode and on the service port, that a plain server omits counters it does not maintain, that a
batched server's counters move with traffic, that nothing listens by default, and — on Linux —
that the prefork parent emits one distinct series set per worker and every worker moves.

## Cost

Nothing is added to any path a request travels, and no worker changes at all. In `--prefork` the
page is rendered from inside the parent's dispatch loop, on the one thread that owns that state,
so there is no shared memory, no snapshot and no atomic. Reads and writes to the scrape socket
are non-blocking and one-shot: a scrape that would block is **dropped** rather than retried,
because losing one sample is cheaper than delaying the server that produced it.
