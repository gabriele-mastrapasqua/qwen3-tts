# Watching a server — a working Prometheus + Grafana example

This directory is a **recommended starting configuration**, not a product. It exists because
"the server exposes metrics" and "somebody can actually see them" are different claims, and the
second one is the one worth shipping evidence for.

What it contains:

| file | what it is |
|---|---|
| `prometheus.yml` | a scrape config pointing at `--metrics-port` |
| `grafana-dashboard.json` | six panels built on the per-worker series |
| `../../tools/observability_up.sh` | downloads and starts both, provisioned, on loopback |

## Fastest path

```bash
# 1. a server with metrics on
./qwen_tts -d qwen3-tts-0.6b --serve 8080 --prefork 4 --prefork-threads 8 \
           --batch-size 8 --metrics-port 9109

# 2. Prometheus + Grafana, provisioned with the dashboard, both on 127.0.0.1
tools/observability_up.sh

# 3. from your laptop, tunnel in -- do not open these ports to the world
ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 user@your-box
```

Then `http://127.0.0.1:3000` → **Qwen3-TTS serving**.

Everything binds to loopback on purpose. A metrics page publishes concurrency, capacity and
build identity; Grafana with anonymous access publishes all of it to anyone who can reach port
3000. On a rented box, a tunnel is the whole security model.

## What the dashboard is built to show

The panels exist to make one class of failure visible, and it is the one a summed view hides:

- **Completed per second, per worker** — `rate(qwen_tts_worker_completed_total[1m])`. A worker
  whose line goes flat while the others carry on is wedged, and the total throughput may barely
  move.
- **In flight** and **saturation** (`inflight / slots`) — where the pressure actually sits.
- **Imbalance** — `max(qwen_tts_worker_inflight) - min(...)`. On a real soak this sits above
  zero even when everything is healthy, because request durations vary; what matters is whether
  it stays there.
- **Rejects per second by reason** — and mind the scope: in `--prefork` these are the parent's
  refusals only. A rejection inside a worker's own queue is invisible from the parent.

## What you must not read off these charts

A `histogram_quantile` or a `rate()` here is an **operations** signal. Acceptance numbers --
TTFA, `safe_play_start`, stall rate, `STREAM_RTF` percentiles -- come from the harnesses in
[`../../docs/serving/cpu-operations.md`](../../docs/serving/cpu-operations.md), measured
client-side over raw samples. The server cannot honestly report a stall: a stall is an event in
a player, and the server does not hold the playback clock.

See [`../../docs/serving/metrics.md`](../../docs/serving/metrics.md) for the full series list
and the reasoning behind what is and is not exported.
