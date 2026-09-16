# AR-1 · Post-P2 architecture review — what this engine should become

Task: AR-1 (read-only review of the frozen P2 checkpoint; produces the AR-2 ordering proposal)

Reviewed HEAD: `8d9ff3de9bb75009d42d43c3875925c16e9e2540` (`docs: close P2 decoder checkpoint`),
branch `feature/x86-amx-vnni-oss`, worktree clean. Reference host and configuration:
GCP c4-standard-24, Xeon Platinum 8581C, 12 physical cores, SMT off, Qwen3-TTS 1.7B INT8,
Design-D INT8 AMX decoder with persistent B packs, 2x6 prefork, engine-owned pool, batch
cap 2, q8, ragged threshold 2. Nothing was implemented, run on the box, or committed by
this review.

## Question

After P2, what is the smallest coherent architecture change that breaks the coupling
between compute efficiency, decode burst size and playback cadence, and in what
dependency order should Codex build it?

## Known facts (measured, tracked)

- Envelope on the current architecture (`.work/p1-cadence-truth-20260907.md` CT-5):
  C2 GOOD (STREAM 0.600/0.623, prebuffer p95 47 ms), C3 GOOD (0.664/0.786, 370 ms),
  C4 MARGINAL (0.793/0.856, prebuffer p95 596 ms, stall@500 25%).
- Cadence law confirmed (CT-1): q8 → q32 moves C4 required prebuffer p95 from ~0.34-0.73 s
  to ~2.2-2.5 s while STREAM_RTF p95 moves 0.83 → 0.94-1.01. Gang joining alone is not
  the cause (`GANG_MIN=64` neutral).
- Decoder cost per call (CT-2): fixed intercept 23-28 ms plus 9.3-9.6 ms per frame at
  q8; at q32 8-13 ms plus 12.5-13.3 ms per frame.
- Talker step (CT-4, 1x6 diagnostic): B1 21.6 ms per frame; B2 23.7 ms per iteration,
  18.0 ms per slot-frame (ratio 1.10, per-slot 0.83x).
- Input length (`.work/p2-input-length-scaling-20260907.md`): at C1 prefill/TTFA grow
  46/80 ms (16 tokens) → 394/432 ms (112 tokens); at C3 an arriving request first waits
  for the predecessor's whole inline prefill (54 → 426 ms), so C3 long-input TTFA p95
  is ~905 ms. Established streams saw a 309 ms inline prefill coincide with a ~235 ms
  larger gap (one pair, CT-3). Long prompts above 512 tokens are rejected, not split.
- Machine utilization at C4: 7.3-7.4 of 12 cores busy in the P2 A/Bs
  (`.work/p2-sq2-direct-quant-20260907.md`), 8.3 core-equivalents in the older chunk
  sweep; on c8a the pool threads ran at ~75 % and the pool was asleep ~25 % of wall
  (`docs/runtime-map-c8a-c4.md`). About 35-40 % of the machine is idle at the
  MARGINAL operating point.
- Decoder structure at HEAD (code inventory, `qwen_tts_speech_decoder.c`): the batched
  path with two active slots is ragged and issues about 102 pool rendezvous per
  8-frame chunk in the conv stack alone: 12 INT8 AMX conv1 panels, 12 FP32 k=1 conv2
  SGEMMs, 40 transposed-conv tap SGEMMs, 8 ConvNeXt SGEMMs, 1 initial conv, 29 snake
  dispatches. Only 12 of ~102 touch AMX. The SQ-1 warm strip applies only to the
  per-slot path (single active slot); it is a no-op for the ragged two-slot call.
- Server/scheduler sources (`qwen_tts.c`, `qwen_tts_server.c`, `qwen_tts_thread.c`) are
  unchanged since the pre-P2 review: one engine thread per worker runs admission, inline
  full prefill, lockstep Talker step, batched CP, inline gang decode and blocking socket
  writes (E5 in `.work/professional-streaming-architecture.md`).
- Talker sequence layout (`qwen_tts.c:1198-1337`): [instruct][role][codec control
  + speaker][text block + tts_eos, all before any codec frame][codec bos]; generation
  attends over the whole text; no per-frame text interleaving; RoPE table 8192.
  `qwen_talker_prefill` already runs from a nonzero `pos0` for prefix-cache hits
  (`qwen_tts_talker.c:1499-1547`), i.e. resumable prefill is mechanically supported by
  the kernel path; what forbids it is `qwen_tts.c:1398` resetting `kv_len` and the
  one-shot `input_embeds` rebuild. The ICL path prefills arbitrary 16-codebook frames
  as an acoustic prefix with the same embedding recipe as generation
  (`qwen_tts.c:1297-1318`, `1679-1689`).
- SQ-3 (P2 checkpoint): dispatch/matrix/addressable shares are 100 % of the measured
  Design-D eligible subset; whole-request `amx_request_wall_share` is UNKNOWN. The
  old ~10-20 % figure is historical context only.

## Unknowns (not measured, must not be treated as facts)

- CP wall per frame at B=1 and B=2 on this host (the only number is the older cost
  map, ~12 ms per frame); the per-frame budget below marks it as an estimate.
- Whether the two-submitter overlap (decoder thread on the engine pool, not a private
  team) helps or starves; never measured since a449f60.
- Exact trained layout of Qwen3-TTS streaming text input; the official inference only
  simulates it.
- Fused residual AMX serving effect: the checkpoint records "positive short serving
  signal" but no tracked table exists.

---

## 1. Executive verdict

The server is limited by **serialization on one engine thread per worker**, not by
decoder arithmetic and not by AMX reach. Every stage of a stream (admission prefill,
Talker step, CP, decoder, socket write) is executed by the same thread in one lockstep
loop with fork-join pool dispatches; at C4 each worker keeps only ~3.7 of its 6 cores
busy, and 35-40 % of the machine idles while established streams wait. Three concrete
consequences follow and explain every P1/P2 number:

1. **Playback cadence** is set by the decode burst: a two-item 8-frame decode
   (~25 + 2 × 76 ms ≈ 175 ms) stops both Talkers, so chunks arrive every
   8 × ~40 ms + ~175 ms ≈ 500-550 ms (measured max_gap p95 554-594 ms) and the ramp
   → q8 transition needs 300-600 ms of prebuffer. Larger quanta amortize the 25 ms
   intercept but multiply the burst (the CT-1 law).
2. **First play** for anything but short text is dominated by full inline prefill
   (46 → 394 ms for 16 → 112 tokens) plus waiting behind another request's prefill or
   an in-flight decode burst; long input at C3 costs ~0.9 s p95.
3. **Capacity** is not compute-bound at C4: the idle fraction, the B ≈ 1 Talker weight
   re-read per worker (B2 already saves 17 % per stream), and ~102 fork-join
   rendezvous per decoder chunk are scheduling and structure costs, not MAC costs.

P2 proved that removing individual memory passes inside the decoder (direct ConvT,
depthwise, warm input, one-row quant) does not move the end-to-end envelope, because
none of them changes the burst, the serialization, or the rendezvous count. The next
work must change **who runs what when**: a cooperative, bounded-quantum engine loop
where decode and prefill are resumable jobs interleaved between Talker steps under a
per-stream playback-lead policy. Strip fusion, fused residual, and single-engine
Talker batching remain valuable, but as capacity work layered on that scheduler.

## 2. P2 retrospective

| experiment | mechanism | observed result | verdict | architectural lesson |
|---|---|---|---|---|
| Warm decoder range slice / direct INT8-A (`QWEN_SD_STREAM_STRIP`, 44c6d49) | Convolve only newly produced columns from per-slot tails; skip the 6-54 discarded context columns | Correct, in the serving reference; per-slot path only, no-op for the ragged two-slot call | 1 (sound, promising) with a scope caveat | Sound saving, but it does not touch the ragged path that carries C4; "reference" status overstates coverage |
| Direct ragged/streaming ConvT (7f3ff3f, 7d336aa) | Accumulate taps into the useful range + carry instead of over-length buffer + copy | Byte-identical; local −2 to −9 % on larger groups; C4 STREAM p95 −0.001, cadence unchanged/worse in sample | 3 (local, not end-to-end) | Removing a copy does not change burst length or rendezvous count (still 40 per-tap SGEMM dispatches) |
| Direct depthwise (cb4be8b) | In-place 7-tap over the global workset, no per-item alloc/copy | Byte-identical; neutral-to-slightly favorable; not defensible | 3 | Same as above; the cost it removed was small relative to the 25 ms intercept |
| Direct warm input (f3637dd, 86cedfa) | No `[tail|input]` materialization; split prefix/suffix fill | Warm ext 4.4-4.9 ms → 0.03 ms locally; 12-request C4: STREAM p95 0.808 → 0.816, prebuffer p95 314 → 439 ms | 3, with a hint of 2 | A split traversal can cost more than the copy it removes; locality is the currency, not bytes |
| Fused residual AMX (b325807 per-slot, b11f991 ragged) | Residual add in the conv2 epilogue; ragged conv2 moves from FP32 SGEMM to INT8 Design-D | Positive short signal (untracked table); not byte-identical on the ragged path | 5 for serving, 1 for mechanism | The only P2 change that reduces a full-size pass AND converts 12 FP32 rendezvous into AMX panels; needs a quality gate (section 10) |
| BLAS-C residual (53fac21, reverted) | GEMM with beta=1 into the residual buffer | Worse C4 tail/cadence | 3-4 for this realization | OpenBLAS with beta=1 on a caller-owned C changes its packing/partition path; FP32 fusion has no structural gain |
| One-row gather+quant (aa90518, reverted) | Quantize each gathered column immediately, never write the FP32 panel | Byte-identical; C4 STREAM p95 0.776 → 0.954, max gap 517 → 1151 ms | 2 (concept sound, realization wrong) | The panel is the SIMD/locality unit; direct quantization must stay panel-wide (batch-aware) or not exist |
| Low-N M split (`.work/amx-c4-cross-request-20260907.md`) | Split high-M, single-panel calls into 96-row AMX tasks | 312 AMX tasks, zero fallback; STREAM p95 0.971 → 1.106 | 4 for this workload | More tasks = more rendezvous and shorter tiles; tile count is not the bottleneck |
| Ragged threshold 8 → 2 (`.work/amx-c4-ragged-threshold-20260906.md`) | Let small-panel calls use the pool | p95 1.0035 → 0.954, prebuffer unchanged (~2.4 s at q32) | 1 for capacity, 3 for cadence | Pool participation helps wall time; it cannot fix a quantum that is 2.56 s of audio |
| Cross-request decoder aggregation (1x12 / batch 4) | Larger ragged N across streams | items=4 formed; STREAM 1.125/1.277 | 2 (context prevented the gain) | Decoder N is already wide at n=1; aggregation only enlarges the inline burst; the probe measured lockstep coupling, not batching |
| Engine-owned decoder pool + partitioned BLAS (a449f60, pre-P2) | One thread team; SGEMMs partitioned on the engine pool | WAV-identical; soak 1.03/1.11 → 1.00/1.06 on c8a | 1 | The one structural change that helped: fewer competing teams. It also makes the two-submitter overlap testable for the first time |
| Ragged panel parallelization (3f7e0df) | Panels claimed from an atomic cursor on the pool | STREAM p95 1.338 → 0.870 | 1 | Parallelism inside a call is the cheap win that existed; it is now exhausted |

Reading across the table: every "3" removed bytes inside a call whose wall is set by
its intercept (25 ms), its ~102 rendezvous, and the fact that the whole call blocks
the Talker. A different realization of DIRECT_QUANT (panel-wide) would still be a "3"
unless it is part of a change that reduces rendezvous or bounds the burst. That is
the mechanism test required by the brief: the retrospective supports (B) in the strip
question below, not (A): partial transformations cannot remove enough, because the
dominant terms are structural.

## 3. Current pipeline map (batched worker, HEAD)

| stage | owner | blocks other streams | preemptible | batching axis, effective B | character | fixed vs variable | belongs to |
|---|---|---|---|---|---|---|---|
| accept/route | parent, least-loaded fd pass | no | n/a | none | trivial | fixed | first play |
| parse + tokenize | reader thread | no | n/a | none | trivial | fixed | first play |
| admission + full prefill | engine thread, inline, `ADMIT_PREFILL` | **yes: every slot stops 46-394 ms** | no (one-shot, `kv_len` reset) | none; bf16 matmat 16-token tiles | compute (bf16) | linear in prompt tokens | first play + continuity |
| Talker step | engine thread, lockstep over B_eff slots, one region per 28-layer step | yes (it is the loop) | between steps only | slots, B_eff 1.1-1.3 at C4 | DRAM-bandwidth bound (~1.5 GB INT8 per step, 21.6 ms at B1) | ~fixed per step, weakly rising with B | continuity + throughput |
| sampling / embed | engine thread, per slot | yes | between slots | none | scalar | small | continuity |
| CP | engine thread + pool region, batched over slots, 16 sequential codebooks | yes | between frames | slots | L3-resident GEMV/GEMM, ~640 barriers per frame-pair | ~fixed per frame | continuity + throughput |
| decoder | engine thread, inline, ragged gang over ready items | **yes: 100-175 ms per call at q8** | no (whole chunk) | ready items (1-2) and panels within a call | glue-bound: ~102 rendezvous, 44 FP32 SGEMMs, 29 snakes, 12 AMX panel calls | 25 ms + 9.5 ms/frame | continuity (burst) + throughput |
| PCM emission | engine thread, 3 blocking writes per chunk | **latent: a slow client stops the worker** | no | none | syscall | fixed | all three |

Coupling summary. One thread, one queue discipline (lockstep), fork-join everywhere:
the pool is idle during every serial section (per-slot attention, sampling, embedding,
decoder glue, BLAS single calls inside regions, socket writes), and the engine thread
is idle during every rendezvous wait. Nothing can be preempted, so the largest job
(long prefill, two-item decode) sets the cadence of everything else.

Per-frame budget per worker at C4 (B ≈ 2), derived from CT-2/CT-4 and the older cost
map; CP is an estimate:

| term | per iteration | share |
|---|---:|---:|
| Talker step (B2) | 23.7 ms | ~37 % |
| CP (B2, estimate) | ~16 ms | ~25 % |
| decoder, amortized (25 + 2 × 8 × 9.5) / 8 | ~22 ms | ~35 % |
| serial glue, sampling, writes | ~2-3 ms | ~4 % |
| total | ~64 ms per 80 ms frame | ρ ≈ 0.80 (measured p50 0.79) |

The same arithmetic explains the 1x12/batch-4 probe: Talker(B4) ~30 + CP(B4) ~20 +
decoder 4 × 12.5 ≈ 50 → ~100 ms per 80 ms frame ≈ 1.25 (measured 1.28) when all of it
sits on one thread. Single-engine batching cannot work without moving decode off the
Talker's critical path; that is a scheduling fact, not evidence against batching.

## 4. Root-cause hierarchy

**First play**
1. Full inline prefill scaling with prompt length (46 → 394 ms), then waiting behind
   the previous request's prefill (C3 delayed wait equals the predecessor's prefill).
2. Waiting behind an in-flight inline decode burst (up to ~175 ms at C4).
3. Ramp first chunk (1 frame: ~35 ms). Not a problem.

**Playback continuity**
1. Decode burst on the Talker thread: chunk period = quantum × (Talker+CP) + burst.
2. Admission prefill hole (309 ms) landing inside established streams' lead.
3. Lockstep coupling: all slots hit their quantum boundary together; gang join makes
   the burst two-item. (Not the primary cause: `GANG_MIN=64` neutral.)
4. Blocking socket writes (latent, unmeasured by the loopback harness).

**Throughput / capacity**
1. Serialization idle: ~35-40 % of cores idle at C4.
2. Talker weight re-read per worker at B ≈ 1 (B2 saves 17 % per stream; B4 more).
3. Decoder glue: 25 ms intercept, ~102 rendezvous, 44 FP32 SGEMMs and 29 snakes per
   chunk (~35 % of the per-frame budget for ~0.25 % of one AMX core's MAC peak).
4. CP's 16 sequential codebook steps (L3-resident; batching across slots is its only
   lever; the official runtime re-feeds the whole sequence and is worse, so no easy
   algorithmic gain).

AMX is deliberately absent from these lists: dispatch and eligibility are solved for
the decoder, Talker/CP never present B ≥ 4 under prefork, and the whole-request wall
share has no valid denominator. Making AMX matter is a consequence of fixing 1-3 under
capacity, not a route to fixing continuity.

## 5. Architecture candidates

**A. Cooperative bounded-quantum engine loop (single thread, resumable jobs).**
Keep one engine thread per worker and the engine pool, but turn the two blocking jobs
into resumable state machines executed in bounded slices between Talker steps:
decode slices (one stage or one column strip at a time, 5-20 ms), prefill slices
(N text positions at a time from `pos0`, ~50 ms per 16 tokens). A per-stream
lead/deadline state decides each iteration which slice runs after the Talker step.
Zero new threads, zero numerical change, output quantum decoupled from decode
quantum. Effect: continuity strongly improved (bursts become slices), first play for
established-stream neighbors protected, TTFA of the new request unchanged or slightly
later, STREAM_RTF unchanged, capacity unchanged (same serialization). Risk: low;
complexity: medium (decoder job state machine, prefill cursor, scheduler).

**B. A plus decoder overlap on the shared pool (two submitters).**
A decoder consumer thread executes the decode slices from A and submits them to the
same engine pool (serialized by `submit_mtx`), so decoder glue and Talker serial
sections overlap. Recovers part of the idle 35-40 %; needs chunky dispatches (a slice
must be one rendezvous, not ~102) or the decoder starves behind Talker regions.
Effect: capacity up (lower ρ), continuity as A, first play slightly better. Risk:
medium (pool priority/starvation, oversubscription by one thread). This is NOT the
rejected c8a "decoder thread" run, which used a private second team on the same six
cores before a449f60.

**C. B plus single engine with global Talker/CP batching (1x12).**
One process, all streams in one Talker region (B up to 4), decode consumed as slices
by B, admission by A. Attacks capacity cause 2 and lets Talker/CP reach matrix shapes
where AMX INT8 is legitimately gated on (B ≥ 4). Effect: capacity up materially at
C ≥ 4 (per-stream Talker cost ~22 → ~10-12 ms); continuity as A/B; first play
protected as A. Risk: high without A and B (this is exactly the failed 1x12 probe);
12-way barrier cost inside regions; one slow client would stall four streams without
OUT-1.

**D. Static core lanes (Talker cores vs decoder cores).**
Rejected for C ≤ 4: the Talker at B ≤ 2 needs the full 12-thread bandwidth roof on this
VM, and the decoder needs a tiny fraction of an AMX core; isolation is obtainable in
B without partitioning. Revisit only if C shows a compute-bound Talker at B ≥ 3.

**E. Full strip executor first (P3 as decoder work).**
Rejected as the first step. The retrospective shows the strip's end-to-end value is
(i) a lower per-frame decoder cost (capacity) and (ii) making small quanta cheap
(intercept). Neither breaks the burst; a fused strip executed inline still stops the
Talker for the whole chunk. The strip becomes the natural *body* of the decode slices
in A/B, where its rendezvous reduction (from ~102 to a handful) is also what B needs.

Comparison on the seven properties the brief asks for:

| property | A | B | C | D | E |
|---|---|---|---|---|---|
| first audio latency-prioritized | yes (prefill slices yield to first-audio deadline) | yes | yes | partial | no |
| established streams protected by lead | yes | yes | yes | partial | no |
| no stage monopolizes the engine unboundedly | yes | yes | yes | no (lanes still lockstep inside) | no |
| batching opportunistic, latency-bounded | n/a (prefork B≤2) | n/a | yes | no | no |
| decoder efficiency without multi-second quanta | via small slices; intercept remains until strip | same + overlap | same | same | yes but inline |
| admissions cannot freeze playback | yes | yes | yes | no | no |
| slow clients cannot block inference | needs OUT-1 | needs OUT-1 | needs OUT-1 (worse blast radius) | needs OUT-1 | needs OUT-1 |

## 6. Recommended target architecture

Primary: **B built on A, with C as the capacity step once A and B hold**, OUT-1 in
parallel from the start. One process per worker today (prefork retained until C),
one engine pool, two submitters (engine thread, decoder consumer), no private teams.

```
            reader threads                              parent (prefork today)
   request ──parse/tokenize──▶ admission queue ──────▶ least-loaded worker
                                     │
   ┌─────────────────────────────────▼──────────────────────────────────────┐
   │ ENGINE THREAD (one per worker) — cooperative bounded-quantum loop       │
   │                                                                          │
   │  each iteration:                                                         │
   │   1. Talker step for all active slots (one region, B_eff)               │
   │   2. sample, CP (batched), embed  → codec frames → per-stream FRAME Q   │
   │   3. scheduler picks ONE side-job slice by deadline/slack:               │
   │        a. decode slice for the stream with least lead (if lead < target) │
   │        b. prefill slice (N positions from pos0) for the admitting request│
   │           if no stream is near underrun, or its first-audio deadline    │
   │           is nearer than the neighbors' underrun deadlines              │
   │        c. otherwise: larger decode slice / maintenance                   │
   │   4. hand completed PCM to the per-stream PCM Q (never write here)       │
   │                                                                          │
   │  per-stream state: delivered_audio, wall since first audio → lead,       │
   │  time_to_underrun, first_audio_deadline, frame Q depth, decode cursor    │
   └──────────────┬───────────────────────────────────────┬───────────────────┘
                  │ decode slices (B: run by a consumer     │ PCM chunks
                  │ thread on the SAME pool, chunky:        ▼
                  │ one rendezvous per slice)        ┌────────────────┐
                  ▼                                  │ OUTPUT WRITER   │
        ┌──────────────────────┐                     │ non-blocking,   │
        │ DECODER JOB (resumable)                    │ bounded per-    │
        │ stage×strip cursor per stream;             │ stream queue,   │
        │ tails/carries already per slot             │ drop/close on   │
        │ body = today's ragged stages, later strip  │ overflow        │
        └──────────────────────┘                     └────────────────┘
```

Component boundaries and state:
- **Frame queue** per stream: codec frames produced by step 2, consumed by the decode
  job. Bound: the lead target plus one quantum; if full, the stream's Talker step is
  skipped that iteration (it is already ahead of playback).
- **Decode job** per stream: cursor over (stage, column strip) of the existing ragged
  or per-slot conv stack, with the slot's tails/carries as the only cross-slice state
  (they already exist). Slice = one stage over the pending columns, later one fused
  strip. The job is preemptible at every slice boundary.
- **Prefill job** per admitting request: cursor `pos0` into the assembled prompt
  embeddings; each slice calls `qwen_talker_prefill` for `[pos0, pos0+N)` (the
  prefix-cache code path already handles nonzero `pos0`), writing KV into the slot
  directly rather than into the shared ctx followed by a copy. Preemptible at every
  N positions.
- **PCM queue** per stream, bounded; writer thread with non-blocking sockets and a
  send timeout; overflow policy = close the stream, never block the engine.

Scheduling policy: earliest-deadline among (underrun deadlines of active streams,
first-audio deadline of the admitting request), with the Talker step always taken for
streams whose frame queue is not full. Lead target: enough to absorb one steady-state
quantum plus scheduler jitter (~2 quanta), not maximum buffering. Output quantum:
emit PCM as each slice completes a whole frame group; quantum grows from 1-2 frames
at startup toward 8 as lead allows (LS-2), never above what the lead can absorb.

Batching policy: opportunistic and bounded. Within prefork: B ≤ 2 as today. Under C:
a Talker step waits at most one iteration for a joining slot; a decode slice may
aggregate ready items only when their leads are both above target; first audio is
never held for width.

Prefill policy: sliced, deadline-aware, on the engine thread (A) and later on the
existing helper thread with LOW pool priority when the engine's slack is negative.
The new request's first-audio deadline (target: TTFA p95 < 500 ms for short/medium)
can preempt neighbors' non-urgent slices but never a slice whose stream is within one
quantum of underrun.

Decoder policy: same numerics as today first (ragged/per-slot stages as slices); then
structural rendezvous reduction (packed ConvT as one op per layer instead of 6-16 tap
SGEMMs; fused residual; snake fused into the strip); then the cache-resident strip
body. Quantum is a scheduler decision, never a decoder decision.

Thread/core ownership: engine thread + pool workers on the worker's mask; the decoder
consumer (B) is one more submitter on the same pool, pool width reduced by one to
avoid oversubscription; output writer is a lightweight thread that never computes.
No static core partition.

## 7. Dependency and order (proposed P3/P4/P5)

The current PLAN order (P3 lead scheduler → P4 topology → P5 output → P6 QL) is
directionally right but under-specified in two places: it treats the decoder as done
after P2 and it leaves OUT-1 behind everything. Proposed:

```
P3  Cooperative bounded-quantum scheduler (architecture A) + OUT-1 in parallel
    P3.a  decoder job state machine (slice = stage over pending columns), PCM
          emission per completed frame group, no numeric change          [LS-3 rescoped]
    P3.b  per-stream lead/deadline state + EDF pick of one side-job per iteration
                                                                        [LS-1]
    P3.c  lead-controlled output quantum replacing the static ramp     [LS-2]
    P3.d  resumable prefill (pos0 cursor, KV written into the slot) as a deadline job
                                                                        [LS-4 + PF-1, new]
    P3.e  OUT-1/OUT-2 non-blocking bounded output (independent, can start first)
P4  Overlap and structural decoder cost (architecture B)
    P4.a  decoder consumer thread on the shared pool, chunky slices, pool priority
    P4.b  rendezvous reduction: packed ConvT one op per layer; fused residual under
          the quality gate; snake fused into the slice                    [SQ-1 body]
    P4.c  cache-resident strip body inside the slice framework           [SQ-1 full]
P5  Single engine and batching economics (architecture C)
    P5.a  1x12 with global Talker/CP batching, decode via P4.a, admission via P3.d
                                                                        [EO-1/EO-2]
    P5.b  AMX INT8 Talker/CP at B ≥ 4 once shapes exist; dynamic threads per stage
                                                                        [EO-3, AMX-0]
P6  Qualification: discover highest GOOD concurrency, long-input dimensions, backends
                                                                        [QL-1/QL-2]
Research arms (gated, not on the critical path)
    RA-1  bounded-window continuation with acoustic history (R3) — serving-only
          re-prefill via the ICL prefix; quality-gated
    RA-2  native streaming-text layout (dual-track) — read the official modeling code
          first; model-supported but layout unverified
    RA-3  PREFILL-Q — deferred behind all of the above
```

Why this order: A is the smallest change that breaks the burst/cadence coupling and
protects established streams from admission, with no numerical or ownership risk. B
needs A's slice structure to exist. C needs B (decode off the Talker path) or it
reproduces the 1x12 failure. The strip needs A's cursor and B's chunky-slice
requirement to have an end-to-end effect; done first it repeats P2. OUT-1 is
independent and cheap, and C is unsafe without it.

## 8. Small falsification experiments (before any large implementation)

| # | assumption | experiment (existing knobs, Tier A, C3+C4 unless stated) | kill criterion |
|---|---|---|---|
| F1 | The machine is serialization-bound, not compute-bound, at C4 | Per-thread CPU from the wave `cores` field and `/proc` ticks per pool thread (already emitted by `serve_procstats`) on the CT-5 config | If workers show > 90 % core utilization at C4, overlap (B) has no room; only kernel speed helps |
| F2 | Small quanta are affordable enough for lead-controlled output | q4 and q2 arms alongside q8 (`QWEN_STREAM_DECODE_CHUNK`/`_BUSY`) at C3/C4 with the playback harness | If q4 raises STREAM p95 above 0.95 at C4 or prebuffer does not fall below q8's, the intercept must be cut (P4.b) before P3.c can go below q8 |
| F3 | Moving prefill off the frame loop protects established streams | `QWEN_PREFILL_HELPER=1` with `QWEN_PREFILL_LOW_MS` on, established-streams + long-arrival scenario (CT-3 shape), C3 | If neighbors' stall_rate@500 and max_gap during admission do not improve versus inline, sliced prefill on the engine thread (P3.d) is the wrong lever and only capacity fixes admission |
| F4 | Two submitters on one pool interleave usefully | `QWEN_DECODER_THREAD=1` with the engine-owned pool at HEAD (per-slot decode, no gang), C4 | Predicted failure: decode latency inflates (~100 dispatches each waiting behind a Talker region) → max_gap explodes while STREAM improves. If max_gap p95 > 800 ms, B requires P4.b (chunky slices) first. If both improve, P4.a is nearly free |
| F5 | Fused residual AMX is numerically acceptable | Golden set (`make test-golden` shapes) with `QWEN_SD_FUSED_RESIDUAL=1` per-slot and ragged; mel-corr, duration, plus per-layer relative RMS of conv2 outputs versus FP32 | mel-corr < 0.99 on any golden item, or audible artifacts in the ear check, or relative RMS > 1e-2 on conv2 outputs → keep default-off |
| F6 | Single-engine Talker batching amortizes at B4 | 1x12 diagnostic, `[ITER]` Talker wall at B=1/2/4 with decode disabled or decoder thread on (no cadence claim) | step(B4) > 2.0 × step(B1) → P5.a demoted |
| F7 | Resumable prefill is mechanically safe | Unit test: prefill in two calls `[0, k)` then `[k, n)` via the existing nonzero-`pos0` path; compare KV and `dec_x` bit-for-bit with one-shot prefill | Any mismatch → P3.d needs kernel work, not just a cursor |
| F8 | Bounded-window continuation is viable (RA-1) | CLI-only: generate window 1; re-prefill `[control][text tail + window 2][bos][last 2-3 s of generated codes]` through the ICL path; listen and compare against one-shot | Audible reset or speaker drift on 3 of 5 samples → RA-1 parked |

F1-F4 and F6 use only existing flags and the current binary; F5, F7, F8 need a test
harness or CLI run, no runtime change.

## 9. Do-not-implement list

- Larger decode quanta for RTF (q16/q24/q32): rejected by the cadence law.
- More AMX tile tasks (low-N M split, synthetic task widening): rejected; tile count
  is not the bottleneck.
- One-row direct gather/quantization: rejected realization; only a panel-wide
  batch-aware form may return, and only inside P4.b.
- BLAS-C (beta=1) residual fusion: rejected.
- Cross-request decoder aggregation for width: decoder N is already wide; aggregate
  only for isolation/cadence and only above target lead.
- A private second thread team for the decoder on the same cores (the c8a +20-50 %
  run): superseded by the shared-pool consumer (P4.a).
- Static core lanes (D) before C shows a compute-bound Talker.
- 1x12 single engine with inline gang decode (the failed probe) before P3/P4.
- Naive sentence splitting as the long-input answer (independent prefills per span,
  no continuity state).
- Appending text KV after codec positions in the current layout: the model was not
  trained on it; RA-2 must read the official streaming layout first.
- BF16 decoder arm or W4 as the P3 fix; PREFILL-Q before the architecture.
- More direct copy-elimination flags in the decoder (P2 showed they are "3"s).

## 10. Quality and qualification gates

An optimization may become default only when all of the following hold on the same
binary, same seeds, same stratified bank, Tier A first then Tier B:

- Correctness: bit-identical WAVs where the numerical route is unchanged; where it
  changes (fused residual ragged, any INT8 route change, RA-1 windows), mel-corr
  ≥ 0.99 and duration within 5 % on the golden set, per-layer relative RMS reported,
  and an ear check on the emotion/clone samples.
- Errors = rejects = timeouts = 0 (unless overload is under test).
- TTFB and TTFA p95 not worse than control by more than 5 % for short and medium
  input; long-input TTFA reported separately and not hidden in a pooled percentile.
- STREAM_RTF p95 not worse than control; ≤ 0.90 preferred at the operating point.
- required_prebuffer p95 and safe_play_start p95 not worse than control; targets
  ≤ 500 ms and ≤ 1 s, preferred ≤ 250-300 ms and ≤ 800 ms.
- stall_rate@500 not worse; → 0 at the operating point; stall_rate@250 reported.
- Admission interference: established streams' stall_rate@500 and max_gap during a
  long arrival not worse than control (the CT-3 scenario becomes a standing test).
- Slow/stopped client scenario: no effect on other streams' metrics (after OUT-1).
- Coalesced-read share reported; runs above ~10 % are diagnostic only.
- Class mix reported; a drifted mix invalidates pooled drift claims, not the run.

## 11. Open unknowns

Facts: everything under "Known facts" above. Estimates: the per-iteration budget in
section 3 (CP term), the ~500-550 ms gap decomposition, the B4 Talker cost, the
decode-latency inflation predicted for F4. Unsupported unknowns: the whole-request
`amx_request_wall_share`; whether the engine thread's serial sections or the pool's
rendezvous waits dominate the idle 35-40 % (F1 splits them); the fused residual
serving gain; whether the official streaming-text layout is a per-position dual-track
interleave that this engine can reproduce (RA-2); the quality of ICL-based window
continuation on this model (F8); whether OpenBLAS partitioned SGEMMs or project-owned
packed ConvT ops are the cheaper way to cut the 40 tap rendezvous.

## 12. Proposed PLAN revision (for review; PLAN.md not edited)

Replace the current P3-P5 blocks with:

- **P3 Cooperative bounded-quantum scheduler (architecture A) + output isolation**
  - LS-3' decoder job as resumable slices with per-frame-group PCM emission (no
    numeric change) — first, it is the skeleton everything else uses.
  - LS-1 per-stream lead/deadline state and EDF side-job selection.
  - LS-2 lead-controlled output quantum (ramp replaced; F2 sets the floor).
  - PF-1 (new, absorbs LS-4) resumable prefill cursor with KV written into the slot;
    admission as a deadline job; F3/F7 first.
  - OUT-1/OUT-2 moved here from P5, independent, can start immediately.
- **P4 Overlap and structural decoder cost (architecture B)**
  - EO-4 (new) decoder consumer on the shared pool with priority; F4 first.
  - SQ-4 (new) rendezvous reduction: packed ConvT per layer, fused residual under the
    F5 gate, snake fused into slices.
  - SQ-1' cache-resident strip body inside the slice framework.
- **P5 Single engine and batching economics (architecture C)**
  - EO-1/EO-2 single engine with global Talker/CP batching over P3/P4; F6 first.
  - EO-3/AMX-0 dynamic per-stage thread width; AMX Talker/CP at B ≥ 4; useful AMX
    wall share measured with a real denominator.
- **P6 Qualification** unchanged (QL-1/QL-2) plus the long-input dimensions.
- **Research arms** RA-1 window continuation, RA-2 streaming-text layout, PREFILL-Q
  (deferred), each gated by the section 10 quality rules and by a CLI-level falsifier
  before any server work.

Rationale in one line: P2 showed that inside-the-call savings do not reach the
player; the burst, the serialization and the admission hole are scheduler facts, so
the scheduler comes first, overlap second, batching third, and the decoder's deeper
fusion rides inside that structure instead of ahead of it.

## Conclusion

What limits the server after P2 is one thread per worker doing everything in lockstep
with unpreemptible bursts, not decoder MACs and not AMX reach. The smallest coherent
change is a cooperative bounded-quantum engine loop with resumable decode and prefill
jobs under a per-stream lead policy (A), with non-blocking output; overlap on the
shared pool (B) then recovers the idle third of the machine, and single-engine
batching (C) is the capacity step that finally gives Talker/CP the shapes where AMX
is worth having. Strip fusion and fused residual are real but belong inside that
structure. C4 GOOD is a plausible outcome of A alone; C5+ needs B and C.

## Next action

AR-2: adopt the P3/P4/P5 ordering in section 12 after review, run F1-F4 and F7 on the
current binary as the first Tier A session (one box session, no code), then start
P3 with LS-3' and OUT-1 in parallel.
