# Task · MT-4 transport boundary

## Question

Can the batched streaming server expose a real response-header event before model
synthesis, and avoid a small-packet Nagle delay, without changing inference or
PCM production?

## Known facts

- The batched server used to send the chunked HTTP header lazily from
  `sink_on_chunk`, together with the first audio chunk. Its client TTFB was
  therefore numerically the same event as TTFA.
- The non-batched `handle_tts_stream` path already sends the header before
  synthesis. This change aligns the continuous/batched path with that boundary.
- `set_client_timeout()` is called on accepted sockets in the batched and
  prefork paths. The change enables `TCP_NODELAY` there; it does not alter
  the async output queue or introduce a second inference path.

## Unknowns

- This Tier-A screen does not establish a long-run throughput or playback
  qualification. It tests event ordering and short-server behavior.
- The synchronous PCM callback still performs its existing blocking writes;
  OUT-1/OUT-2 remains the independent slow-client isolation feature.
- A server-side per-audio-chunk flush timestamp is not added here. The
  client marks remain client-observed, although this screen measured zero
  coalesced reads.

## Files/functions inspected

`qwen_tts_server.c` (`set_client_timeout`, `sink_next_job`, `sink_on_chunk`,
`sink_on_done`, `handle_tts_stream`), `tests/serve_parallel_wave.py`, and the
existing OUT/P0 transport evidence.

## Implementation

Commit `a0bef4f`:

- sets `TCP_NODELAY` on accepted client sockets where the platform exposes it;
- sends the `200` chunked response header in `sink_next_job` for an admitted
  synchronous streaming job, before the continuous engine loop starts;
- retains the old lazy-header fallback in `sink_on_chunk` for jobs that did not
  publish a header, and marks a failed early header write as cancellation.

The async writer remains unchanged and continues to own its own header write.

## Evidence

### Local gates

- clean native `make blas`: PASS;
- `make test-selftest`: PASS, both dispatched and forced-fallback paths;
- `make check-flag-registry`: PASS, 199/199;
- `python3 tools/check_repo_integrity.py`: PASS with the pre-existing
  Makefile reference warning for `docs/emotion-seeds.md`;
- `python3 tools/check_plan.py`: PASS;
- `git diff --check`: PASS.

The patch is confined to server socket/header handling; no model arithmetic or
PCM conversion code changed, so the numerical/audio path is unchanged by
construction. A golden audio run was not required for this transport-only edit.

### AMX server path screen

The source was committed and clean-built as `a0bef4f` with `SIMD=amx`. The
binary SHA-256 prefix was `7b6e453e75f30720`. The reference host was the
single-socket/NUMA Xeon 8581C deployment with twelve physical CPUs online and
SMT off; workers were `2x6` on masks `0-5` and `6-11`. The model was 1.7B
INT8 with Design-D INT8 AMX, engine decoder pool, ragged threshold 2, q8,
batch cap 2, and the promoted warm strip. No profiler or census was enabled.

The short true-wave screen used the five-entry short bank, three waves at C1
and C3, and zero coalesced reads:

| C | TTFB p50/p95 ms | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | required prebuffer p95 ms | safe start p95 ms | stall@100/@250/@500/@1000 | errors/rejects |
|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0.4 / 0.6 | 82.6 / 84.2 | 0.546 / 0.555 | 0.567 / 0.581 | 13 | 96 | 0% / 0% / 0% / 0% | 0 / 0 |
| 3 | 1.0 / 61.4 | 158.2 / 167.0 | 0.682 / 0.883 | 0.713 / 0.944 | 381 | 540 | 56% / 11% / 0% / 0% | 0 / 0 |

The result is a boundary proof, not a C3 qualification. At C3 the header is
observed well before first audio while the streaming and playback values remain
within the expected short-screen envelope. The run's AMX flags and topology
were printed by the server; the decoder path was not changed.

A separate low-overhead lifecycle trace on the same build recorded, for each
batched stream, `write_attempt` immediately after `admitted` and
`write_complete` milliseconds later, before the later `ttfa_after_admit` work.
This confirms the event ordering in the server's monotonic clock domain. The
trace is diagnostic and is not used as a KPI run.

## Conclusion

**PROMOTE as a transport/metric-truth fix.** MT-4 is implemented and proven
on the real continuous server path. TTFB is now a distinct header event for
batched synchronous streams, and TCP_NODELAY is enabled on accepted sockets.
It does not claim to solve slow-client blocking; use the independently gated
async output path for that question.

## Next action

Keep the output-queue default decision separate from MT-4. Continue with the
already ordered P3/P4 architecture work; do not reopen decoder or AMX kernel
experiments because this transport boundary is now explicit.
