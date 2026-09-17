#!/usr/bin/env python3
"""metrics_watch.py — watch the serving metrics page while load runs, and check it adds up.

Point this at --metrics-port during a soak or a wave. It samples the page on a fixed
interval and prints, per worker, what moved since the previous sample. It is a CHECK, not a
dashboard: at the end it reports three things a scraper cannot survive being wrong.

  monotonic     no counter ever went backwards.  Prometheus reads a decrease as a process
                restart and silently discards the interval, so a counter that resets under
                load produces a graph that is quietly missing traffic.
  conserved     completed <= dispatched for every worker, always.  A worker cannot finish
                work it was never given; if it appears to, the two counters are being
                written from different places and one of them is wrong.
  balanced      every live worker received something.  One idle worker while the others
                saturate is the failure per-worker series exist to expose -- and the one a
                summed view hides.

Usage:
    tools/metrics_watch.py --url http://127.0.0.1:9109/metrics --duration 120 --interval 10

Run the load separately (serve_parallel_wave.py, load_test.py, the soak): this only watches.
"""
import argparse, re, sys, time, urllib.request

SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)$')
LABEL = re.compile(r'(\w+)="([^"]*)"')


def scrape(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    out = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        labels = dict(LABEL.findall(m.group("labels") or ""))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out[(m.group("name"), labels.get("worker"), labels.get("reason"))] = value
    return out


def workers(sample):
    ws = {k[1] for k in sample if k[1] is not None}
    return sorted(ws, key=lambda w: int(w) if w.isdigit() else w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:9109/metrics")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds to watch")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between samples")
    ap.add_argument("--timeout", type=float, default=3.0, help="per-scrape HTTP timeout")
    a = ap.parse_args()

    try:
        first = scrape(a.url, a.timeout)
    except Exception as e:                                   # noqa: BLE001 - report, don't trace
        sys.exit(f"cannot scrape {a.url}: {e}")
    if not first:
        sys.exit(f"{a.url} returned no samples — is the server running with --metrics-port?")

    ws = workers(first)
    print(f"watching {a.url} · {len(ws)} worker(s) · every {a.interval:g}s for {a.duration:g}s")
    print(f"{'t':>6}  {'worker':>6}  {'inflight':>8}  {'+dispatched':>11}  {'+completed':>10}")

    prev, t0, samples = first, time.time(), 1
    monotonic, conserved, seen_work = True, True, {w: False for w in ws}
    regressions = []

    while time.time() - t0 < a.duration:
        time.sleep(max(0.5, a.interval))
        try:
            cur = scrape(a.url, a.timeout)
        except Exception as e:                               # noqa: BLE001
            print(f"  scrape failed: {e}")
            continue
        samples += 1
        t = time.time() - t0

        for key, value in cur.items():
            if key[0].endswith("_total") and key in prev and value < prev[key]:
                monotonic = False
                regressions.append(f"{key[0]}{{worker={key[1]},reason={key[2]}}} "
                                   f"{prev[key]:.0f} -> {value:.0f}")

        for w in workers(cur):
            infl = cur.get(("qwen_tts_worker_inflight", w, None), 0.0)
            disp = cur.get(("qwen_tts_worker_dispatched_total", w, None), 0.0)
            done = cur.get(("qwen_tts_worker_completed_total", w, None), 0.0)
            d_disp = disp - prev.get(("qwen_tts_worker_dispatched_total", w, None), disp)
            d_done = done - prev.get(("qwen_tts_worker_completed_total", w, None), done)
            if disp > 0:
                seen_work[w] = True
            if done > disp:
                conserved = False
            print(f"{t:6.0f}  {w:>6}  {infl:8.0f}  {d_disp:11.0f}  {d_done:10.0f}")
        prev = cur

    live = [w for w in workers(prev) if prev.get(("qwen_tts_worker_up", w, None), 1.0) > 0]
    idle = [w for w in live if not seen_work.get(w)]
    balanced = not idle

    print(f"\n{samples} samples")
    for w in workers(prev):
        print(f"  worker {w}: dispatched "
              f"{prev.get(('qwen_tts_worker_dispatched_total', w, None), 0.0):.0f}"
              f"  completed {prev.get(('qwen_tts_worker_completed_total', w, None), 0.0):.0f}")
    for reason in ("all_workers_full", "fd_dispatch_failed", "queue_full", "queue_timeout"):
        v = prev.get(("qwen_tts_rejected_total", None, reason))
        if v:
            print(f"  rejected[{reason}]: {v:.0f}")

    print(f"  monotonic: {'yes' if monotonic else 'NO'}")
    if regressions:
        for r in regressions[:10]:
            print(f"    went backwards: {r}")
    print(f"  conserved: {'yes' if conserved else 'NO — completed exceeded dispatched'}")
    print(f"  balanced:  {'yes' if balanced else 'NO — idle worker(s): ' + ','.join(idle)}")

    ok = monotonic and conserved and balanced
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
