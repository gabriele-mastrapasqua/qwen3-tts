# Professional continuous streaming — architecture rationale and plan basis

Task: MT-1 CT-1 SQ-1 LS-1 EO-1 OUT-1 QL-1 (the streaming redesign; this addendum is
the evidence base for every group P0-P6 in PLAN.md)

## Question

Which architecture lets this CPU engine start quickly, feed a real 1x player without
repeated starvation, keep realtime headroom, isolate streams from one another, and only
then maximize sustainable concurrency and cost efficiency? And why did the previous
objective ("maximize concurrency while STREAM_RTF p95 < 1 and TTFA is acceptable") stop
explaining serving behavior?

## Known facts

Reference host and configuration (all numbers below unless stated): GCP c4-standard-24,
Xeon Platinum 8581C, one socket, 12 physical cores, SMT off, CPUs 0-11; Qwen3-TTS 1.7B,
INT8, decoder Design D INT8 AMX with persistent B packs, two prefork workers x six
threads, engine-owned decoder pool, decoder batching on, server batch cap 2.

- Best trusted short C4 control (ragged pool threshold 2, chunk 32):
  STREAM_RTF p50/p95 0.900/0.954, TTFA p95 ~565 ms, zero errors/rejects/timeouts,
  zero-buffer required prebuffer p95 ~2.4 s, stall_max ~2.0 s
  (`.work/amx-c4-ragged-threshold-20260906.md`).
- Chunk sweep, short wave, zero-buffer prebuffer p95: chunk 8 → 0.450 s, 12 → 0.757 s,
  16 → 0.785 s, 24 → 0.927 s, 32 → 1.222 s, while STREAM p95 moved only 0.919 → 0.880.
  Mixed bank at chunk 32: underrun p50/p95 2.15/2.55 s
  (`.work/amx-c4-chunk-sweep-20260906.md`).
- Low-N M split: 312 real AMX row tasks, zero fallback, short C4 STREAM p95
  0.971 → 1.106. Rejected. Single-process 1x12 with batch cap 4 formed real
  `items=4` worksets and measured STREAM p50/p95 1.125/1.277. Not promoted
  (`.work/amx-c4-cross-request-20260907.md`).
- Per-worker effective batch at C4 is about 1.1-1.3 (aggregate ~2.3-2.5); the Talker
  and CP therefore run mostly as B=1 GEMV and never reach the AMX INT8 gate (B >= 4)
  (`.work/p1-4-amx-runtime.md`, `.work/amx-native-epic.md` section 9).
- C4 cost map, per request, inclusive regions: admission 97 ms, Talker prefill 239 ms,
  Talker decode 461 ms, CP decode 254 ms, decoder 631 ms of which conv stack 604 ms
  (`docs/reference-gcp-c4-standard-24.md`). The decoder is roughly 40 % of engine
  wall; Talker+CP+prefill roughly 60 %.
- Inline admission prefill stalls every established slot for 108-240 ms per admission,
  about 10 % of wall at C4 soak (`docs/runtime-map-c8a-c4.md` section 7, item 1).
- Frame = 1920 samples at 24 kHz = 80 ms. Chunk 32 = 2.56 s of audio per stream.
- The old "~14 % AMX MAC share" census is invalidated and must not be cited
  (`.work/amx-native-epic.md` section 8). The corrected pre-Design-D census: INT8 AMX
  0 %, VNNI matmat ~12.3 %, INT8 GEMV ~10.6 %, BF16 AMX ~2 %, decoder ~76.6 % of MACs
  and outside the dispatcher. Design D later moved the decoder conv1 work to real
  `TDPBSSD`.

## Unknowns

- Whether client receive marks track server flushes closely enough: the harness reads
  through Python HTTP buffering, so prebuffer/stall remain "client-observed" quantities.
  MT-4 now sends the batched-stream header at admission and sets `TCP_NODELAY`; the
  synchronous PCM callback still uses blocking writes. See the MT-4 addendum.
- The decoder's fixed per-call intercept and per-frame slope have not been measured
  directly; they are inferred from the chunk sweep (see Evidence, E3).
- Whether a single-engine Talker step at B=2..4 actually amortizes the weight read on this
  VM; never isolated from decode coupling.
- C2 and C3 envelopes on this host: no rows exist (C1 and C4 only).

## Files/functions inspected

- `qwen_tts.c`: `qwen_tts_serve_continuous` (frame loop, lockstep over slots), chunk
  ramp `decpos → target` (1, 2, 2, 4, 4, then `QWEN_STREAM_DECODE_CHUNK`), gang
  `must/join/leader` selection and the inline
  `qwen_speech_decoder_decode_streaming_batch` call, `ADMIT_PREFILL` on the loop thread,
  `QWEN_ADMIT_M1`, `QWEN_TTFA_PRIORITY`, the opt-in decoder thread `dec_pool_t` and the
  opt-in prefill helper.
- `qwen_tts_server.c`: prefork least-loaded fd routing, reader threads, `scheduler_main`,
  `sink_on_chunk` → `send_pcm_chunk` (three blocking `write(2)` per chunk from the engine
  thread, no send timeout, no non-blocking mode).
- `qwen_tts_speech_decoder.c`: `sd_stream_batch_body`, ragged conv stack, `rag_conv1d_amx`
  panel worker, `rag_convt` (per-tap BLAS plus serial scatter), snake dispatch, tails.
- `qwen_tts_thread.c`: engine pool, `submit_mtx`, region depth, nested-dispatch rule.
- `tests/playback_sim.py`, `tests/soak_client.py`, `tests/serve_parallel_wave.py`,
  `tests/serve_soak.py`: STREAM_RTF, TTFA/TTFB, zero-buffer prebuffer/underrun/stall.

## Evidence

### E1. What the metrics measure

`STREAM_RTF = (t_done − t_first_chunk) / (audio delivered after the first chunk)`. It is a
mean rate over the stream interval. `prebuffer_s = max_i[(t_i − t_first) − audio held
before chunk i]` is a max-lateness statistic over the same timeline. They disagree
exactly when delivery is quantized: a stream can have STREAM_RTF 0.90 and still deliver
2.56 s of audio every ~2.3 s. **STREAM_RTF < 1 does not prove that a player starting at
first audio never stalls.** It remains the capacity metric; it is not a streamability
qualification. `docs/serving-operations.md` section 5 currently states the opposite in
its STREAM_RTF row and calls prebuffer/underrun "diagnostic, not a KPI"; that
interpretation is superseded by this addendum and must be corrected under MT-1.

### E2. The cadence law (chunk geometry, not compute, explains the 2.4 s)

A chunk of C frames can be delivered only after C Talker+CP steps plus its decode, so
consecutive deliveries are about C × w apart, where w is the wall per frame
(w = ρ_f × 80 ms). Between deliveries the player consumes C × w of audio while holding
only the lead accumulated so far, and lead grows at (1 − ρ) per second of audio. A
stream can absorb a quantum of C frames without stalling only if

    lead ≥ C × w  ≈ ρ_f × (chunk audio)

At ρ ≈ 0.9 lead grows 10 % of audio produced. The ramp delivers 13 frames (1.04 s) before
the first 32-frame chunk, accumulating ~0.1-0.2 s of lead; the first 32-frame chunk then
lands ~2.1-2.4 s after the previous delivery. Prediction versus measurement:

| chunk | audio quantum | predicted ρ_f × quantum − lead | measured prebuffer p95 |
|---:|---:|---:|---:|
| 8 | 0.64 s | ~0.4-0.5 s | 0.450 s (short wave) |
| 32 | 2.56 s | ~2.1-2.4 s | 1.22 s (short wave), 2.37-2.55 s (mixed bank) |

The remaining tail comes from the inline gang decode (up to B × chunk frames in one
uninterruptible call while every Talker stops) and from inline prefill admission
(108-240 ms). Compute insufficiency contributes nothing while ρ < 1. Consequences:
chunk 32 is an RTF artifact and must not be promoted; `safe_play_start ≈ TTFA +
prebuffer ≈ 3.0 s` at C4/chunk 32, which fails any interactive budget regardless of RTF.

### E3. The loop that has to be broken

Decoder fixed per-call cost → pressure toward large chunks → large audio quanta →
bursty delivery → multi-second prebuffer. From the chunk sweep, the 3 ms/frame difference
between 8- and 32-frame quanta implies a fixed cost per decoder call of roughly 30 ms
(intercept) and a variable cost on the order of 15-20 ms per frame per stream (slope).
Both are estimates to be measured under CT-2. Sources of fixed and near-fixed cost read
from the code: about 41 pool rendezvous per call (12 residual conv1 panel dispatches plus
29 snake dispatches), about 110 BLAS calls (44 transposed-conv taps with serial
scatter-add, 12 k=1 convs, transformer and projections), fp32 im2col materialization of
roughly 1 GB per 32-frame chunk per stream across the 12 residual convs, a separate
quantization pass, about 88 M sine evaluations in snake, per-item tail save/restore, and
the ragged front end (VQ, transformer, ConvNeXt) at N = frames × items.

### E4. AMX accounting (kept as four separate quantities)

| metric | estimate | reading |
|---|---|---|
| amx_dispatch_share | ~100 % of residual conv1 calls; 0 % of Talker/CP calls at B < 4 | dispatch is solved |
| amx_matrix_mac_share | ~55 % of request MACs on `TDPBSSD` (72 % of decoder MACs × decoder ~76 %) plus ~2 % BF16 prefill | eligibility is largely solved |
| amx_addressable_mac_share | ~70 % (adds k=1 convs, transposed convs, pre/initial conv); Talker/CP at B < 4 are GEMV and never addressable | ceiling of any kernel work |
| amx_request_wall_share | ~10-20 % (decoder ~40 % of engine wall × ragged-conv share × ~52 % of ragged-conv wall in tiles) | Amdahl cap: infinite AMX speed saves at most ~15 % wall |

One 32-frame chunk is ~78 GMAC of conv stack per stream, ~31 GMAC/s at realtime, a
small fraction of one AMX core's peak. N in the residual blocks is already 1024n to
61440n columns at n = 1, so tile occupancy is not the problem and cross-request
aggregation cannot improve it. The decoder is glue-bound: data movement, preparation,
barriers and transcendental work around the tiles. That is why the M split and the batch-4
probe regressed: they added dispatch, barriers and burst size to a stage whose tile time
is a minority of its wall. The way to raise useful AMX wall share is to remove the
surrounding glue, not to manufacture tile tasks.

### E5. Pipeline ownership and burst sources (default batched server)

| stage | owner / batching axis | blocks audio | burst source |
|---|---|---|---|
| admission + prefill | engine thread, inline, per request | all streams, 108-240 ms | yes |
| Talker step | engine thread, lockstep over B_eff slots, one region per step | yes | B=1 ↔ B=2 cost cliff |
| CP | batched over slots, 16 sequential codebook steps | yes | small |
| decoder | engine thread, ragged over ready items, inline gang | yes, and stops all Talkers | dominant: B × chunk frames per call |
| emission | engine thread, blocking socket writes, no send timeout | a slow client stalls every stream of the worker | latent correctness defect |

### E6. Worker model

Prefork exists as a workaround for lockstep coupling and pool width, not as an AMX
design. It gives the decoder nothing (its N is already wide) and forecloses the only
large Talker/CP lever: the Talker at B ≈ 1 is DRAM-bound (~22 ms per step at B ≈ 1.2,
~1.5 GB of INT8 weights per step) and two workers read the same weights twice per
frame-time against one memory roof. The failed 1x12/batch-4 probe tested weight
amortization together with inline gang decode of 4 × 32 frames, inline prefill stalling
four streams, and 12-way barriers; it measured the coupling, not the amortization, and
is not evidence against single-engine batching. The earlier decoder-thread run that
measured +20-50 % STREAM ran a second thread team on the same cores before the decoder
moved to the engine pool (a449f60); slice-interleaved decode through the same pool is a
different design and is untested. A global cross-process decoder service would pay IPC
to aggregate matrices that need no aggregation; its only value would be isolation and
cadence control, obtainable in-process.

### E7. Candidate architectures (ranked by end-to-end streaming benefit, then risk)

1. **Lead-controlled cadence scheduler.** Per stream, estimate playback lead
   (audio delivered − wall since first audio, plus a safety margin) and fire a decode when
   the pending frames at the measured wall per frame would otherwise exhaust the lead;
   the quantum is whatever the lead affords, small at startup, growing with accumulated
   (1 − ρ). Admission prefill becomes a deadline job scheduled only when every
   established stream's lead exceeds the prefill cost. Kill: with chunk 8 and gang join
   disabled at C4, prebuffer p95 does not follow ρ_f × quantum.
2. **Small-quantum decoder (streaming strip executor).** Process each residual block in
   cache-resident column strips (snake → conv1 → snake → conv2 → residual per strip),
   quantize directly from the strip into the INT8 A panel without an fp32 im2col, run the
   transposed convs as a packed panel op instead of per-tap BLAS with serial scatter, and
   cut rendezvous per call from ~41 to a handful. Kill: measured tile time already
   > 70 % of decoder wall, or fixed intercept < 10 ms and slope < 8 ms/frame.
3. **Single-engine Talker/CP batching with bounded decoder slices.** One engine, all
   streams in one Talker region so weights stream once per step; decode executed as
   bounded slices between Talker steps on the same pool, no second team; prefill on the
   low-priority helper. Kill: `[ITER]` step wall at B=2 > 1.6 × step at B=1 in one process.
4. **Asymmetric core lanes** (Talker/CP lane, decoder lane with its own AMX units, one
   lead-ordered decode queue). Value is isolation, not width; likely loses at C ≤ 4
   because the Talker at B ≤ 2 needs the full 12-thread bandwidth roof. Only if 3 shows
   the Talker becomes compute-bound at B ≥ 3.
5. **Non-blocking per-stream output.** A correctness requirement, not a performance lever;
   rides along with 1 or 3.

### E8. Provisional qualification envelope

Provisional engineering targets, not external promises; they become gates only after MT-1
validates the harness semantics.

| dimension | mandatory | preferred | strong |
|---|---|---|---|
| correctness | parity PASS, errors = rejects = timeouts = 0 (unless overload is under test) | | |
| TTFB | measured independently of TTFA | < 100 ms | |
| TTFA p95 | reported | < 500 ms | ≤ 700 ms acceptable only for materially better continuity |
| STREAM_RTF p95 | < 1 | ≤ 0.90 | ≤ 0.85 |
| required_prebuffer p95 | reported | ≤ 500 ms | ≤ 250-300 ms |
| safe_play_start p95 | reported | ≤ ~1 s | ≤ ~750-800 ms |
| stall_rate@500ms | → 0 at the operating point | | stall_rate@250ms → 0 |
| established streams under admission | no stall induced by a new arrival at the 250-500 ms buffer | | |
| slow/stopped client | unrelated streams unaffected | | |

Classification per concurrency: GOOD (all preferred met), MARGINAL (mandatory met,
preferred missed), NOT STREAMABLE. Capacity is the highest GOOD concurrency with useful
margin, discovered, not prescribed. If C3 is GOOD and C4 is MARGINAL, C3 is the operating
point.

### E9. Benchmark workflow

Tier A, development: C3 + C4 (C1/C2 only to isolate a mechanism), short stratified bank,
one architectural question per experiment, always with the full startup/RTF/playback
metric set. Tier B, qualification: stationary stratified workload with realistic mixed
lengths, sustained arrivals, sufficient duration, per-request playback simulation with
fixed buffers, fairness, admission under established load, slow-client scenario,
cancellation/disconnect, zero errors/rejects/timeouts. Class-mix drift must not be
allowed to invalidate latency-drift interpretation (fix the schedule, report the mix).

### E10. Historical results, classified

| result | class |
|---|---|
| C4 soaks before `db8fba6` (no playback fields) | throughput-valid, TTFA-valid, RTF-valid; playback continuity unknown |
| C4 threshold-2 control 0.900/0.954, chunk sweep, mixed-bank chunk 24/32 | RTF-valid, TTFA-valid; zero-buffer playback diagnostic only; qualification superseded by the streamability definition |
| chunk 32 "best" | RTF artifact; NOT a production winner (E2) |
| low-N M split | rejected, negative, keep visible |
| 1x12 / batch 4 probe | diagnostic-only; does not isolate Talker/CP batching (E6) |
| decoder thread +20-50 % STREAM (c8a) | valid for a second thread team; superseded for the same-pool sliced design |
| AMX census "14 % MAC share" | invalidated, do not cite |
| AWS/GCP reference docs (c8a, c8i, c4) | hardware/backend performance evidence; continuity claims there rest on zero-buffer prebuffer, fixed-buffer stall rates not measured |
| `.work/amx-ragged-scheduler-review-3f7e0df.md` | correctness review, still valid |

### E12. MT-1/MT-2 result (2026-09-07): receive-marker semantics and metric definitions

Transport audit, read from `qwen_tts_server.c` and CPython 3.10 `http.client`:

- Batched server: MT-4 sends the `200` header from `sink_next_job` immediately after
  admission, before the continuous engine loop. Non-batched `handle_tts_stream` already
  sends the header before model work. The harness stamps TTFB and TTFA independently and
  reports `header_to_audio_ms`; the short AMX server screen observed zero coalesced reads.
- Each PCM chunk remains three blocking `write(2)` calls (size line, payload, CRLF); MT-4
  sets `TCP_NODELAY` on accepted sockets but does not add `writev`, `MSG_MORE`, a synchronous
  send timeout or non-blocking mode. There is no server-side PCM accumulation on the
  synchronous path; the terminator and `close()` remain synchronous from the engine thread.
  Chunk boundaries are the engine's decode quanta.
- Client mark = the instant `HTTPResponse.read1()` returns one HTTP chunk (or part of one)
  to the harness, after the GIL is reacquired. `read1` never spans two chunks and does not
  wait for the chunk's trailing CRLF (consumed lazily at the next call). A late reader finds
  several chunks in the 8 KiB buffer or the kernel queue and returns them microseconds
  apart, so N server emissions can appear as N marks at one instant: the earlier ones are
  stamped LATE, and prebuffer/stall are upper bounds on server lateness. Each mark now
  carries the time blocked in the read; a value under 1 ms is a `coalesced read` and the
  share is reported per run (`coalesced reads x%`). The soak runs one client process per
  stream; the wave uses threads in one interpreter and is more exposed.
- Nagle: with three writes per chunk the 2-byte CRLF can be held for a delayed ACK on a
  real network, which shifts to the NEXT chunk's inter-arrival gap (the CRLF is read
  lazily). All harnesses use loopback, where this is not observable. `tests/load_test.py`
  decodes framing by hand and reads the CRLF before stamping, so its per-chunk stamps
  differ by exactly that CRLF wait; it remains the C=1 TTFA oracle.
- Verdict: transport buffering does not destroy cadence fidelity; it biases marks late by
  a measurable, per-run-reported amount. MT-4 is now implemented: the batched header is a
  pre-synthesis event and accepted sockets use `TCP_NODELAY`. A per-chunk server flush
  timestamp remains optional; the async output feature, not MT-4, owns slow-client isolation.

Metric definitions now implemented once in `tests/playback_sim.py` and consumed by
`tests/soak_client.py`, `tests/serve_parallel_wave.py`, `tests/soak_drift.py`:

| metric | definition (per request, marks `(t_i, bytes_i)` from request send, audio `d_i`) |
|---|---|
| TTFB | status line + headers parsed by the client |
| TTFA | `t_0`, first non-empty audio chunk |
| STREAM_RTF | `(t_done − t_0) / Σ_{i>=1} d_i` |
| required_prebuffer | `max(0, max_{i>=1} [(t_i − t_0) − Σ_{j<i} d_j])` |
| safe_play_start | `max_{i>=0} (t_i − Σ_{j<i} d_j)`, direct scan; equals TTFA + required_prebuffer per request, aggregated p50/p95 over requests |
| zero-buffer player | starts at `t_0`, pauses only when empty: `underrun_total`, `stall_max`, `stall_count` |
| fixed-buffer player @B | starts when B s of audio is buffered (or the stream ended), re-buffers B after an underrun: `stall_ms@B`, `stalls@B`, `stall_max_ms@B`, `start_delay_ms@B`, B in {100, 250, 500, 1000}; @0 reproduces the zero-buffer player |
| stall_rate@B | share of requests with `stalls@B > 0`; `prebuffer_le_rate@B` = share with required_prebuffer <= B (a time-based start delay of B is stall-free exactly for those) |
| max_gap | largest inter-arrival gap after `t_0`; `gap_ratio_max` = max gap / chunk audio |
| coalesced reads | reads that returned in under 1 ms (already-queued data) over all reads |

Tests: `python3 tests/playback_sim.py` (known-answer self-test), `python3
tests/test_playback_sim.py` (smooth, bursty quanta, late-first-chunk, repeated-gap,
degenerate, summaries), `python3 tests/test_soak.py` (CSV layout, drift analyzer with and
without the new columns). CSV columns appended: `safe_play_start_ms, header_to_audio_ms,
max_gap_s, coalesced_reads, stall_ms_at_{100,250,500,1000}, stalls_at_{...}`; older CSVs
still analyze (absent columns read as n/a, never NaN).

### E11. Documentation that needs semantic correction (under MT-1, non-destructive)

- `docs/serving-operations.md` section 5: STREAM_RTF row ("below 1.0 a player starting
  at the first chunk never stalls") and the sentence demoting prebuffer/underrun to
  diagnostics.
- `docs/BENCHMARKING.md` sections 7-8 (terminology and canonical server metrics): add the
  playback metrics and the two-tier workflow.
- `docs/reference-aws-c8i-8c-amx.md` and `docs/reference-aws-c8a-16c-vnni.md`: "never
  stall"/"gapless" statements are zero-buffer-based; annotate, do not rewrite.
- PLAN objective wording ("STREAM_RTF p95 < 1 with useful margin") is replaced by the
  envelope above.

## Conclusion

The server is not compute-starved at C4; it is cadence-broken by construction. Fixed
chunk geometry, inline gang decode, inline prefill and a glue-bound decoder form one
loop, and every local AMX or threshold knob has been tuning inside it. The redesign
order is dictated by dependencies: metric truth first (nothing can be gated on numbers
whose transport semantics are unverified), then cheap cadence discriminators on the
current binary, then the small-quantum decoder and the lead-aware scheduler (which need
each other), then ownership/topology once decode bursts no longer couple streams, then
output isolation, then discovery of the highest GOOD concurrency and backend comparison
under the same harness. Multi-precision work (BF16, W4) stays valid but waits behind
this sequence unless it removes a proven current bottleneck; INT8 is the serving
reference.

## Next action

MT-1 (harness semantics, per-request `safe_play_start`, fixed-buffer stall rates, TTFB
independent of TTFA) with self-tests and no runtime change. Then CT-1: the chunk-quantum
discriminator at C3 and C4 on the current binary (chunk 8, chunk 32, chunk 8 with gang
join disabled), with `[DECODE] dur_ms` and `[ITER]` traces on, which simultaneously tests
the cadence law, yields the decoder intercept/slope regression, and sizes the Talker
B=1→2 step amortization.
