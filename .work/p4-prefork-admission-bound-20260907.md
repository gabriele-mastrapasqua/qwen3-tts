# P4 prefork admission boundary — 2026-09-07

Task · P4 admission correctness under prefork overload

Question · Can an explicit zero-queue policy reject a request at the prefork parent,
instead of allowing it to wait in the kernel listen backlog where the child queue
deadline cannot observe it?

Known facts · The production-shaped prefork parent tracks each worker's active slot
count. Before this change it stopped polling the listening socket when every worker
was full. The child assigns `t_recv` only after it receives and parses the connection,
so time spent before that point is not covered by `--queue-timeout-ms`. The default
queue/backlog behavior must remain unchanged.

Unknowns · This is an overload-rejection fix, not a capacity increase or a complete
lead/deadline scheduler. It does not prove C4/C5 qualification, bound accepted-job
service time, or eliminate inline prefill and slow-client coupling. The product
choice of an operating `--max-queue` value remains open.

Files/functions inspected · `qwen_tts_server.c`: `qwen_tts_serve_prefork`, the
parent `poll`/`accept` loop, `qwen_tts_server_set_limits`, `jq_push`, and child
`sink_next_job`; `tests/serve_parallel_wave.py` output and the prefork startup/
counter logs.

Evidence ·

- Implementation commit: `3db3b768f42fc4a97be88cc17a64792988ca6fb3`.
- Clean AMX build identity: source/build tag `3db3b76:clean`; underlying binary
  SHA-256 `17d14f7bd39b9df0c1d2ea5bcb376af26b34ae136c6506e0dc04b1b22f884b79`.
- Host: GCP `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA, 12 physical
  CPUs online, SMT off, 2x6 prefork. AMX capability and Design-D flags were
  enabled; no intrusive census was used for this admission diagnostic.
- Diagnostic: true simultaneous C5 wave, two waves, batch cap 2 per worker,
  q8, ragged threshold 2, engine decoder pool, fused residual and Design-D INT8.
  `--max-queue 0` was the only admission treatment. Receive coalescing was 0%.
- The exact committed binary accepted 8 requests and rejected 2 at the parent.
  Intermediate prefork counters report `dispatched=8 rejected=2`; startup also
  prints the explicit zero-queue parent-rejection policy. The rejected requests
  received HTTP 503 rather than waiting for a child queue slot.
- Among the 8 accepted requests: TTFB p50/p95 `68/187 ms`, TTFA `423/562 ms`,
  STREAM_RTF `0.764/0.788`, TOTAL_RTF `0.805/0.831`, and zero accepted-request
  execution errors/timeouts. Playback diagnostics were required-prebuffer
  `300/314 ms`, safe-play-start `737/819 ms`, max-gap p95 `588 ms`, and
  stall@500 `0%`; the zero-buffer diagnostic still reports an underrun p95 of
  `314 ms` and is not a server starvation proof.
- The prior control with the same overload shape and old parent behavior allowed
  the excess connections to remain pending: it showed no rejects and TTFB/TTFA
  p95 around `4470/4635 ms`. A child queue-timeout arm likewise kept its child
  queue/pre-service delay low while the client-facing tail remained multi-second.
  This is consistent with the wait occurring before the child's `t_recv`, not in
  the job queue.

Conclusion · **PROMOTE as a bounded admission-safety mechanism, keep the default
queue policy unchanged.** When `g_cfg_max_queue == 0` and the explicit historical
`QWEN_QUEUE_UNBOUNDED` override is absent, the parent continues polling the
listener even with no free worker slot. `accept` then finds no eligible worker and
returns `503 all workers at capacity`. For the default grace queue and the explicit
unbounded A/B override, the old listener-poll behavior is preserved.

This makes the documented zero-queue behavior true and prevents overload latency
from being hidden in the parent backlog. It rejects work; it does not make the
accepted-stream capacity larger. The test is a bounded WAVE/DIAGNOSTIC, not a
qualification run, and the wrapper hash printed by the harness is not used as the
source identity; the server startup build line and independent binary hash above
are authoritative.

Next action · Keep the broader LS-4 deadline-aware admission task open. If a
production profile adopts fail-fast overload, run it as an explicit policy arm and
measure established-stream continuity separately from rejected-request behavior.
The next architectural work remains lead/deadline protection and output isolation,
not another AMX kernel change.
