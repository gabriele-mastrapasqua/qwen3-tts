# AMX ragged scheduler review — 3f7e0df

Task · Preserve the independent review state for the ragged decoder AMX panel
scheduler before server qualification.

Question · Is commit `3f7e0dfa5832216e6c26a5b66f57e6f6a6c0bf67` safe to qualify on
the streaming server?

Known facts · The independent read-only review verdict is **SAFE TO CONTINUE
QUALIFICATION**. The reviewed implementation is the parallel ragged-panel
scheduler at that exact code commit.

Unknowns · The effect of decode-chunk size; whether further matrix aggregation
is needed; and direct underrun/starvation events, which the current SOAK runner
does not count.

Files/functions inspected · Decoder ragged panel dispatch and join/fallback
logic; decoder-pool wrapper; existing serial/parallel server output checks.

Evidence · The review verified disjoint panel output ownership, per-worker
scratch/TLS accumulator semantics, join-before-bias/tail update, exact
recompute fallback, no nested engine-pool BLAS dispatch, and unchanged
portable/VNNI/Arm paths. Serial and parallel server WAVs were already
byte-identical.

Conclusion · **SAFE TO CONTINUE QUALIFICATION**.

LOW / PERF (deferred) · A worker can allocate scratch after the atomic panel
cursor is effectively drained. Do not apply only an `atomic_load` early return:
if optimized later, use a claim-first scheme that claims real work before large
scratch allocation and preserves allocation-failure full-recompute semantics.

LOW / DEFERRED / PRE-EXISTING · The private decoder pool has no complete
after-fork reset if it had been started before fork. The current production
path does not appear to make this reachable. If fixed later, reinitialize the
pool synchronization state (mutex/condition/team state), not only counters.

Targeted future tests (non-blocking) · Forced fragmented-ragged parity;
allocation-failure recompute; BF16 serial/parallel parity; private-pool
variant; forced AMX-reject parallel fallback.

## C4 qualification result — 2026-09-06

The clean code commit was exercised through the 1.7B production streaming
server on the canonical Emerald Rapids host: 2 workers × 6 threads, SMT off,
concurrency 4, decoder batching on, Design D INT8 on, BF16 decoder off, decode
chunk 8, and the engine-owned decoder pool. The closed-loop SOAK ran for five
minutes after a 30-second warm-up, with five 60-second windows and no intrusive
profiler/census.

The valid run completed 136 requests with 120 post-warm-up KPI samples and 16
audio probes. Errors, queue rejects, queue timeouts and server request timeouts
were zero. Post-warm-up pooled KPI:

| metric | p50 | p95 |
|---|---:|---:|
| TTFA | 215.2 ms | 496.4 ms |
| STREAM_RTF | 0.9419 | 1.0350 |

Per-window `STREAM_RTF p95` was `0.996, 0.997, 1.096, 1.013, 1.075`.
The server confirmed 24 persistent INT8 B packs (19.3 MB), decoder batching,
and engine-pool ownership; final dispatch was 171 with zero rejects. Resource
samples reached about 10.1 CPU-core equivalents; aggregate peak RSS/PSS was
about 19.3/9.45 GB.

Conclusion · **C4 NOT QUALIFIED**. TTFA and queue health were acceptable, but
`STREAM_RTF p95=1.035` fails the hard realtime target and has no useful margin.
The per-class drift check also failed. The chunk sweep and C5/C6 are deferred.

Configuration correction · A preliminary attempt passed `QWEN_SD_POOL=engine`.
The code accepts only `1` or `q`, so it resolved to the private decoder pool and
is discarded as qualification evidence. The corrected run used `QWEN_SD_POOL=1`.

Next action · Keep this as the clean engine-pool baseline. Before another code
change, run a bounded diagnostic for panel/batch attribution and direct
starvation/underrun visibility; do not start chunk sweep or higher concurrency
until C4 has useful realtime margin.
