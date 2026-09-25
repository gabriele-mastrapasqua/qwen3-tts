# Cancel on disconnect, session books and fault injection

Task: CD-1, CD-2, CD-3, CD-4, CD-5, CD-6, CD-7, CD-8, CD-9

## Question

1. When a client disconnects mid-request, does the model stop working for it, on every
   serving path (batched slot, single-job clone, plain server; stream and WAV)?
2. Does every request end exactly once in the server's own counters, on every path,
   and can the server prove it (a conservation invariant anyone can read)?

The method follows the companion ASR server's fault work (2026-09-25): every zombie
case first proves it is discriminating, model work after the disconnect is measured
from server counters, and each case checks the books, a neighbour, and memory.

## Known facts (read in the code at `e391ec5`, before this work)

- FACT: `QWEN_CANCEL_ON_DISCONNECT` defaulted OFF (`qwen_cancel_on_disconnect`, env
  must be `1`). With it off, the batched synchronous writer set `client_gone` on a
  failed write but never `cancelled`; `sink_cancelled` returned 0, the slot generated
  to EOS into a dead socket.
- FACT: the engine asks `sink->cancelled` once per slot per frame (`SAMPLE_SLOT`);
  `CANCEL_SLOT` waits for the decoder lane (`dec_wait_idle`) before freeing the slot's
  stream state. That ordering is kept unchanged.
- FACT: the single-job path (`handle_tts_stream` / `handle_tts`: requests with
  `instruct` or `voice_design` on the batched server's clone worker, and every request
  on the plain non-batched server) had no cancellation at all: `stream_http_callback`
  ignored its `write()` results and returned 0, and `qwen_tts_generate` had no
  per-frame check. The flag did not reach it.
- FACT: the disconnect detector used `POLLRDHUP` on Linux, i.e. a FIN (client shut
  its write side) counted as gone.
- FACT: terminal counters `term_ok/client_gone/timeout/rejected` were bumped from two
  hooks. Queue-full and queued-too-long refusals, invalid bodies, the scheduler's
  failure drain, the clone-worker failure and the whole single-job path touched none.
  No conservation invariant existed.
- FACT: the synchronous sockets had no send timeout: a client that stops reading holds
  the writer (the scheduler thread or a decoder lane) in `write()` indefinitely.

## Unknowns

- Linux behaviour of the detector (RST as POLLERR|POLLHUP, FIN as POLLIN + EOF) is
  expected from the kernel semantics but was not run here; macOS only (CD-6).
- Prefork: the books are per worker in the shared page and the parent renders them per
  worker; `/v1/health` stays one worker's scope. Not exercised on this Mac (prefork is
  Linux-only).
- Memory under a long fault soak (hours, many workers): not measured.

## Files/functions inspected

`qwen_tts_server.c`: `qwen_cancel_on_disconnect`, `peer_hung_up` (removed),
`write_all_or_gone`, `stream_http_callback`, `handle_tts`, `handle_tts_stream`,
`handle_connection`, `reader_main`, `sink_next_job`, `sink_on_chunk`, `sink_cancelled`,
`sink_on_reject`, `sink_on_done`, `scheduler_main`, `single_worker_main`,
`qwen_metrics_request_done`, `qwen_metrics_render`, `handle_health`,
`stream_output_*`. `qwen_tts.c`: `qwen_tts_generate` frame loop, `decoder_thread_fn`,
`qwen_tts_serve_continuous` (`SAMPLE_SLOT`, `CANCEL_SLOT`, `RECORD_FRAME_AND_EMBED`),
`qwen_tts_clone_for_worker`. `qwen_tts_compose.c`: `qwen_compose_render_stream`.

## Evidence (2026-09-25, Apple M1 8 cores, 0.6B CustomVoice bf16, `make blas`)

All runs: `tests/test_server_faults.sh` (`make test-server-faults`), batched server
`--batch-size 4 --max-queue 1`. DEVELOPMENT SIGNAL on macOS, not a qualification.
The measure is `frames_generated` in `/v1/health` (new: one per Talker frame, all
paths), read just before the disconnect and again once the server is quiet.

### Zombie cases: frames generated after the disconnect

Every case discriminating: 36-41 s of audio still pending at the disconnect (bound
>= 10 s), measured against an undisturbed reference run of the same request.

| case | before, default (flag off) | before, flag on | after, default (on) | after, `=0` |
|---|---|---|---|---|
| rst-mid (batched stream) | 497 (39.8 s) | 0 | 0 | 497 |
| fin-mid (batched stream) | 497 (39.8 s) | 0 | 8 (one chunk) | 497 |
| rst-first (first audio bytes) | 517 (41.4 s) | 0 | 0 | 517 |
| rst-mid-single (instruct, clone) | 454 (36.3 s) | **454** | 1 | 454 |
| fin-mid-single | 454 (36.3 s) | **454** | 11 (one chunk) | 454 |
| rst-wav (`/v1/tts`) | 493 (39.4 s) | 1 | 0-1 | 493 |

RESULT: before, the default ran every abandoned request to its end (the whole
remaining utterance, every case); the flag alone fixed the batched slot but not the
single-job path. After, a reset stops within one frame, a FIN within one written chunk
(the next write draws the client kernel's RST); `=0` reproduces the old zombie exactly.

### Semantics

- RESULT half-close-ok: `shutdown(SHUT_WR)` right after the request, then read: the
  stream completes, 1989120 bytes, sha identical to the reference, booked completed.
- RESULT stopped-reader (client `SO_RCVBUF` 4 KB, stops reading after 1 s): the request
  ends as client_gone after 21.7-22.8 s; 112 frames were generated after the reader
  stopped (the socket buffers filling), 505 were pending. The blocked write is what
  the 5 s send timeout bounds; while it blocks, the synchronous scheduler serves nobody.
- RESULT neighbours: a 12.5 s stream running while three others die around it (RST,
  FIN, RST on first audio) is byte-identical to its unloaded reference (batching pinned
  with the test-batch-invariance pins, so only the lifecycle could change it).

### Session books

- RESULT: every case moves exactly `sessions +1` and one outcome, active returns to 0,
  `balanced` true, 0 anomalies. abort-loop: 24 aborts (RST / FIN / first audio, stream,
  WAV and single-job) = `client_gone +24`.
- RESULT mixed workload on a server with a 3 s budget, `--batch-size 2 --max-queue 1`:
  3 completed + 2 RST + 2 WAV over the budget (503, discriminating) + 3 stream holders
  over the budget + 1 queue-full 503 + 1 invalid body moved the books by exactly
  sessions 12 = completed 3 + client_gone 2 + timeout 5 + rejected 2, identical deltas
  in `/v1/health` and `/metrics`, `books_balanced 1`, active 0.
- Memory: see abort-loop below.

### Memory (abort-loop)

- FACT: `ps` RSS on macOS is dominated by the mmapped weights and moves by hundreds of
  MB on its own; a first 1.10x RSS bound failed at 1.39x without being a leak.
- RESULT: 10 rounds of 6 aborts (RST / FIN / first audio; stream, WAV, single-job) vs a
  no-abort control of the same mix, `phys_footprint`: 3.4-4.3 GB with aborts,
  3.5-4.6 GB control, no trend; live MALLOC_LARGE regions 119 at round 1 and round 10
  (the growth sits in freed blocks the allocator keeps). No evidence of a per-abort leak.
- DECISION: the suite keeps a GROSS bound (last/first round <= 1.25x; final run
  4064 -> 4591 MB, 1.114x). A fine leak needs the control comparison or a Linux RSS run.

### Gates run on the final tree (macOS)

- `make blas`; `./qwen_tts --self-test` PASSED (0 failed); `--dispatch-map` resolves;
  `make check-flag-registry` PASS (232/232); `tools/check_repo_integrity.py` and
  `tools/check_plan.py` PASS.
- `make test-server-faults`: 40 invariants OK, 0 failed.
- `make test-serve`, `test-serve-openai`, `test-serve-parallel`, `test-serve-repro`: PASS.
- `make test-serve-batch`: FAIL, pre-existing and unrelated: `tests/serve_batch.sh`
  posts with `curl -d` (form Content-Type), which the server's HTTP pre-check has
  refused with 415 since `9eb1233`; every response is a 229-byte error.
- Not run here (stopped to hand over): `test-serve-stream-batch`,
  `test-serve-continuous`, `test-batch-invariance`, `test-golden`,
  `tests/cancel_correctness.py`.

## Decisions taken in this work

- DECISION: `QWEN_CANCEL_ON_DISCONNECT` defaults ON; `=0` restores the old behaviour for
  A/B only, and the server announces which one runs at startup.
- DECISION: gone = a reset or socket error (per-frame `poll(POLLIN)` +
  `recv(MSG_PEEK)`), or a write that failed (EPIPE, ECONNRESET) or timed out. A FIN
  alone is NOT gone: HTTP/1.1 permits a half-close after a complete request, and the
  half-close case proves such a client is served. `POLLRDHUP` is no longer used.
- DECISION: with cancellation on, every client socket gets `SO_SNDTIMEO` =
  `QWEN_STREAM_OUTPUT_SEND_TIMEOUT_MS` (default 5000).
- DECISION: a session is a POST to a synthesis endpoint that passed the HTTP
  pre-checks; an invalid body is a session `rejected`, so both server modes count alike.

## Conclusion

The zombie was the default: on every path, a client that left kept the model
generating for the rest of the utterance, and the single-job path could not be
stopped even with the flag. It now stops within one frame on a reset and within one
chunk on a FIN, on every path, the half-closing client is still served, and every
request ends exactly once in books that balance in both health and metrics.
VERDICT: KEEP on this branch; PROMOTE after the Linux run (CD-6).

## Next action

1. CD-6: run `make test-server-faults` and `tests/cancel_correctness.py` on a Linux
   box (x86 and Arm), including `--prefork 2` for the per-worker books.
2. CD-7 (decision): a FIN on a WAV request cannot be told from a half-close until the
   final write, so a client that closed completely during a WAV still costs the whole
   generation (bounded by `--max-request-seconds`). Options: accept; treat EOF as gone
   for WAV only (what nginx and Go's net/http do for any request); send interim 1xx.
3. CD-8: the single-job path ignores `--max-request-seconds` (the budget is enforced
   only in `sink_cancelled`); its queue bound is computed from the batched occupancy.
4. CD-9: a stopped reader blocks the synchronous batched writer for up to the send
   timeout, stalling every slot of that worker; the async writer
   (`QWEN_SERVER_ASYNC_OUTPUT`) does not, but is default-off.
