# AR-1 implementation audit against frozen P2

Date: 2026-09-07

Frozen implementation: 8d9ff3de9bb75009d42d43c3875925c16e9e2540

Branch: feature/x86-amx-vnni-oss

Reviewed material: ar1-post-p2-architecture-review-20260907.md, PLAN.md,
professional-streaming-architecture.md, post-p2-streaming-research-agenda.md,
P1/P2 evidence, and the runtime at the frozen commit.

Scope: read-only adversarial implementation audit. No benchmark, runtime change,
PLAN change, or production implementation was performed.

## 1. Executive verdict

AR-1 is directionally useful and can be used as input to AR-2 only with explicit
corrections. The central observation is supported: the current server couples
admission, per-frame Talker/CP work, decoder delivery, and socket output through
one scheduler loop per prefork worker, with blocking pool rendezvous between
serial sections. That is a credible explanation for cadence sensitivity and for
the difference between useful AMX kernel work and serving behavior.

Several AR-1 statements are stronger than the code/evidence permits:

- the engine thread is not the executor of every kernel; it is the pool submitter
  and waiter for many regions;
- 7.3-7.4 core-equivalents of utilization do not prove that idle cores are
  available for useful overlap or that decoder arithmetic is irrelevant;
- 102 is not a count of 102 pool rendezvous;
- prefix-cache nonzero position is not a resumable prefill cursor;
- q2/q4 changes the size of a decoder call, not preemption inside the decoder;
- same-pool overlap has not been demonstrated, and the code does not
  automatically reserve one pool worker for a decoder consumer;
- B2/B1 Talker evidence supports a bandwidth opportunity, not a global
  single-engine implementation conclusion.

Recommended disposition:

AR-1 SAFE TO USE FOR AR-2: YES WITH CORRECTIONS

The corrections below should be treated as constraints on AR-2, not as a request
to reopen P2 or to replace the AR-1 architecture wholesale.

## 2. Claim-by-claim corrections

### 2.1 Root cause: engine serialization

Classification: VERIFIED BY CODE/EVIDENCE, with MISSING IMPORTANT CONSTRAINT and
one OVERSTATED conclusion.

Verified:

- qwen_tts_serve_continuous owns one engine/scheduler loop per prefork worker.
- Inline admission calls qwen_tts_generate and therefore performs full prompt
  construction and Talker prefill before installing a slot
  (qwen_tts.c:2800-2844, 3005-3027).
- The loop performs active-slot packing, codec head, sampling, batched CP,
  decoder delivery, optional second admission, and the next Talker step in a
  fixed sequential control loop (qwen_tts.c:3075-3307).
- In the normal inline decoder arm, a decoder call and its sink callback happen
  before the next Talker step. This makes a due decoder burst a real critical
  path for that worker (qwen_tts.c:3101-3147, 3229-3279).
- The server sink writes chunk framing and PCM synchronously. The three
  write_all_or_gone calls are in the callback path
  (qwen_tts_server.c:1497-1530). A slow socket can therefore hold whichever
  thread invokes the callback.
- A prefork worker has its own batch slots, engine loop, and pool. A request in
  another worker is not a candidate for the same runtime batch, although
  read-only inherited weights may share physical pages.

Required correction:

The phrase “every stage is executed by the same thread” is too literal. The
engine thread owns the sequencing and usually submits pool jobs; qwen_parallel
executes some chunks on the caller and some on pool workers, then the caller
waits for completion (qwen_tts_thread.c:493-569). The correct statement is:
every stage is controlled and serialized at the worker-loop boundary, and the
engine thread cannot advance that stream or schedule its next stage while an
inline decoder/prefill/write or pool rendezvous is outstanding.

The “not decoder arithmetic and not AMX reach” conclusion is OVERSTATED. The
following narrower claim is supported:

- decoder Design-D eligibility and AMX execution for the measured eligible
  ragged Conv1 shapes are proven;
- whole-request AMX wall share is still UNKNOWN;
- the decoder includes VQ/transformer work, ConvT, depthwise work, snakes,
  FP32 k=1 projections, epilogues, allocation/copy work, and final output work
  outside the INT8 AMX panel path.

Thus decoder glue/serialization is a demonstrated bottleneck candidate, but
decoder arithmetic, memory traffic, and non-AMX work cannot be removed from the
capacity model. More overlap could expose a bandwidth or cache bottleneck rather
than convert all measured idle time into useful throughput.

The observed 7.3-7.4/12 core-equivalent utilization is VERIFIED as a P2
measurement. Its causal interpretation is only PLAUSIBLE BUT REQUIRES
EXPERIMENT. It is compatible with serial scheduler sections, pool wait,
short/imbalanced jobs, memory stalls, and request arrival shape. It does not
by itself show that the unused cores can safely run another full decoder team.

AMX accounting must retain its denominator. P2 supports:

| quantity | frozen P2 status | correct interpretation |
|---|---|---|
| amx_dispatch_share | 100% of the measured Design-D eligible subset | Not 100% of decoder or request dispatches. |
| amx_matrix_mac_share | 100% of the eligible subset | The eligible subset is already selected by the current path/gate. |
| amx_addressable_mac_share | 100% of the eligible subset | Not a whole-request AMX coverage claim. |
| amx_request_wall_share | UNKNOWN | No clean whole-request AMX wall denominator was recorded. |

Consequently the AR-1 sentence that AMX is absent from the root-cause hierarchy
is too broad. It is reasonable to say that AMX reach is solved for the tested
eligible panel subset; it is not reasonable to infer whole-decoder or
whole-request AMX wall coverage from those three 100% ratios.

### 2.2 “About 102 pool rendezvous per q8 chunk”

Classification: INCORRECT as phrased; VERIFIED BY CODE/EVIDENCE as an
operation-inventory count after renaming it.

The arithmetic inventory in AR-1 can be reconciled for the nominal two-item
ragged chain as:

| inventory item | source-level count | what it actually means |
|---|---:|---|
| upsample ConvT tap SGEMMs | 40 | 16+10+8+6 calls in the four upsample blocks |
| ConvNeXt work | 8 | 4 kernel-2 ConvT taps plus 4 pointwise MLP SGEMM calls |
| initial Conv1d | 1 | one kernel-7 ragged projection |
| residual Conv1 | 12 | four blocks times three residual blocks; AMX-eligible call sites, not necessarily 12 tiles |
| residual Conv2 | 12 | four blocks times three 1x1 projections when the fused path is off |
| snake calls | 29 | four upsample snakes, 24 residual snakes, one final snake |
| total call-site inventory | 102 | operation/call sites for this path, not pool barriers |

The count is path- and shape-dependent. Fused residual removes/replaces the
12 residual Conv2 control operations; direct ConvT changes the implementation
behind the 40 tap call sites; disabled or small snake work may execute inline.

A pool rendezvous is narrower:

- an engine-level decoder invocation is a blocking call boundary, not itself a
  pool rendezvous;
- rag_conv1d_amx submits one sd_pool_run for an eligible Conv1d call when its
  panel count meets QWEN_SD_RAG_MIN_PANELS; the worker then drains multiple
  panels from an atomic cursor;
- qwen_sd_sgemm submits through qwen_parallel only when BLAS ownership, shape,
  thread, and nesting conditions pass; otherwise it calls cblas_sgemm directly;
- qwen_snake_activation submits only when channels*length reaches its
  QWEN_SNAKE_MIN_WORK threshold and the caller is not already a single-thread
  or nested path;
- calls made while qwen_parallel_active are executed inline to avoid nested
  dispatch.

Therefore the exact number of pool submissions/barriers is not recoverable from
the 102 source inventory alone. It depends on n_panels, threshold, shape,
QWEN_BLAS_OWN conditions, nesting, and whether work is direct or fused. The
correct label for the exact per-chunk pool-rendezvous count is UNKNOWN without
the matching path counters. For the nominal fused-off control chain, the listed
sites provide an upper bound of at most 102 potential pool submissions; the
actual count can be lower, and the fused path changes that upper bound. This is
a valid bound, not an observed range. AR-2 must report separate operation call
counts, pool submissions, and pool wait time.

### 2.3 Resumable / incremental prefill

Classification: INCORRECT for “mechanically supported”; PLAUSIBLE BUT REQUIRES
EXPERIMENT for a new fixed-prompt state machine; MISSING IMPORTANT CONSTRAINT
for appended text.

The current nonzero position is a prefix-cache mechanism, not a general
resumable prefill API:

- qwen_tts_generate allocates and fills the complete input_embeds array for the
  complete sequence before prefill (qwen_tts.c:1198-1337);
- the server explicitly sets ctx->prev_prefill_len = 0 and invokes the complete
  qwen_tts_generate during inline admission
  (qwen_tts.c:2800-2806);
- qwen_talker_prefill derives pos0 only from a matching prefix-cache entry
  (qwen_tts_talker.c:1499-1528);
- for a prefix hit it copies cached prefix K/V and computes n_new positions, but
  still allocates/uses full-sequence prefill scratch and completes all layers
  before returning (qwen_tts_talker.c:1452-1493, 1521-1547);
- the general generation path sets ctx->kv_len = delta_start, then either
  executes remaining positions one by one or invokes qwen_talker_prefill for
  the complete remaining prompt (qwen_tts.c:1368-1425).

The mechanisms must be separated:

| mechanism | current status |
|---|---|
| A. Chunked execution of one logically fixed prefill | Not exposed. Possible future design, but requires a cursor and persistent per-layer/intermediate state. |
| B. Pause and resume that prefill | Not supported by current API. No prefill stage/layer/token cursor is retained across a return. |
| C. Append new text after codec generation has started | Not established and not a mechanical continuation. Current layout places text and EOS before codec BOS/frame generation; appending after acoustic positions changes the model sequence semantics. |
| D. Stateless re-prefill/reconstruction | Current fallback-like possibility, but it recomputes and does not solve the intended admission interference. |

Talker KV state exists after a completed prefill and per-slot Talker KV state is
copied into the batch slot. CP state is initialized/advanced during frame
generation, not exposed as a partially completed prefill state. Decoder state
has its own K/V, latent, VQ pad, and causal tail/carry fields, but those are
post-generation streaming state and do not make prefill resumable.

AR-2 must not treat prefix cache, pausing a fixed prompt, and appending text
after generation as the same operation. The first is implemented reuse; the
second needs a runtime state machine; the third is a model/layout and quality
question.

### 2.4 Bounded decoder slices and preemption

Classification: MISSING IMPORTANT CONSTRAINT and OVERSTATED if “q2/q4 is
preemption” is implied.

The smallest natural decoder unit at this HEAD is a complete call for one
stream/new_frames or one ragged batch call. The server selects when that call
fires, with first/early targets of 1, 2, 4, and then the configured chunk
(qwen_tts.c:3110-3127). A smaller target changes the caller-level quantum; it
does not make the decoder yield inside that call.

The streaming decoder runs the whole chain before returning:

- qwen_speech_decoder_decode_streaming_st enters sd_stream_st_body and then
  conv_decoder_forward_streaming (qwen_tts_speech_decoder.c:2452-2470,
  2288-2441);
- the ragged path concatenates the active items and runs the complete
  ConvNeXt, initial Conv, upsample, residual, final snake, and final Conv chain
  before scattering audio (qwen_tts_speech_decoder.c:3541-3723);
- per-stream tails/carries are updated at stage boundaries, but the state
  structure has no stage cursor, current signal ownership, ragged cursor, or
  resumable downstream output state (qwen_tts.h:293-320);
- temporary signal/intermediate buffers are local to the call and are freed at
  the end or on error.

Safe current boundaries are therefore after a completed decoder call and after
the associated per-stream state/tail update. A residual block or strip is not
automatically a safe yield point merely because it is a visible loop iteration.
True cooperative slicing would require explicit persistent state for the
current stage, signal buffers, ragged offsets, carries/tails, output ownership,
error/fallback state, and final scatter. AR-2 should call q2/q4 “smaller
non-preemptible calls” until that state machine exists.

### 2.5 Same-pool decoder overlap

Classification: PLAUSIBLE BUT REQUIRES EXPERIMENT, with MISSING IMPORTANT
CONSTRAINTS.

The current code supports a technically testable two-submitter arrangement, but
it does not prove useful overlap:

- the optional server decoder consumer has its own queue/thread and cloned
  decoder context;
- that consumer can invoke the decoder and submit work to the engine pool;
- qwen_parallel permits concurrent callers, but P.submit_mtx serializes job
  publication and completion; the second submitter waits while the first
  submission holds the mutex (qwen_tts_thread.c:507-567);
- qwen_parallel_active prevents nested pool submission, so a pool worker runs
  nested decoder work inline rather than creating a second nested team;
- the engine pool width is configured independently. Creating a decoder
  consumer does not automatically reduce the pool by one. The consumer is an
  additional caller thread and can compete for CPUs, memory bandwidth, and
  cache;
- decoder callbacks/writes run on the decoder consumer in that mode, but the
  consumer still shares the same pool and decoder weight/read traffic.

This differs materially from the old private-team regression, so that old result
must not be copied as a direct verdict. It remains relevant as a warning about
team contention and bandwidth. Conversely, “recover the idle third” is not
evidence-backed: idle cycles may be pool-wait, memory-stall, short-job, or
critical-section cycles and may not be exploitable by another decoder job.

Smallest safe falsification experiment, not run in this audit:

- current 2x6, SMT off, engine pool, q8, threshold 2, same C3/C4 workload;
- compare inline decoder with QWEN_DECODER_THREAD=1, keeping all other controls
  fixed;
- use only existing low-overhead TTFA/decode/playback telemetry and a clean
  build, with separate sequential runs.

Kill the overlap hypothesis for that configuration on any correctness/error,
reject, timeout, established-stream underrun, or material max-gap regression;
also kill it as a serving win if STREAM_RTF p95 does not improve while CPU
contention or total wall increases. A positive result must show both a server
KPI improvement and no playback-safety regression; pool occupancy alone is not
success.

### 2.6 Single-engine Talker/CP batching

Classification: VERIFIED BY CODE/EVIDENCE for an existing within-worker
opportunity; PLAUSIBLE BUT REQUIRES EXPERIMENT for the global architecture.

Verified Talker behavior:

- qwen_batch_proj_q selects per-slot matvec when B==1, and a native/generic
  matrix path when B>1 (qwen_tts_talker.c:1910-1964);
- qwen_batch_talker_step_ragged already compacts active slots and uses the
  batched implementation when more than one slot is active;
- per-slot KV caches are stored in the batch object and the batched step takes a
  pos_arr, so different request progress positions are representable
  (qwen_tts_talker.c:2250-2275 and surrounding batch state code);
- the measured B2/B1 step ratio is a real bandwidth signal: B2 was about 1.10
  times the B1 iteration and about 0.83 times the B1 per-slot cost.

Verified CP behavior:

- the batched CP routine handles active masks and computes the same codebook
  index across active slots;
- codebooks g=1 through 14 are sequential because each codebook consumes the
  previous code for that slot, but different slots remain independent within a
  given g (qwen_tts_code_predictor.c:1429-1577);
- a single active slot falls back to a context-swapped scalar CP path, while
  multiple active slots use the batched representation.

Limits:

- the current prefork server has one batch/state owner per worker. A global
  ready set across prefork workers cannot be obtained by changing B alone;
- a single process/global engine would need ownership of per-request Talker KV,
  CP KV, positions, RNG/sampling state, active masks, admission and output
  routing. Existing per-slot storage reduces the data-layout risk but does not
  solve process ownership or scheduling;
- B2/B1 does not establish B4 scaling, does not isolate every CP term, and does
  not prove that a larger batch beats additional cache/bandwidth contention;
- batching requests at different Talker positions is supported by pos_arr, but
  batching CP work still requires the same codebook step across selected slots.
  It cannot merge autoregressive codebook indices across g.

The correct conclusion is: there is a measured bandwidth opportunity and a
promising existing per-worker representation; a one-engine/global ready-set
architecture remains an experiment after decode/output coupling is controlled.

## 3. Verified pipeline and blocking map

| boundary | actual owner at frozen HEAD | what is computed | blocking effect |
|---|---|---|---|
| accept/route | parent/readers and scheduler input | fd/job admission | does not share the worker batch until installed |
| inline admission | scheduler/engine thread | complete input embedding build and full Talker prefill | blocks that worker; established slots cannot advance |
| helper admission, if explicitly enabled | helper thread with cloned context | same full prefill, then K/V/hidden copy into slot | removes some prefill execution from engine thread, but install/copy and queue waits remain; not current control |
| Talker step | scheduler submits/executes batch work | active-slot Talker projection/attention/FFN | loop cannot move to CP/decode until region returns |
| CP | scheduler plus engine pool region | one frame, 16 codebook stages | sequential per codebook index; pool rendezvous/serial work |
| inline decoder | scheduler calls full stream or ragged decoder | VQ/transformer/conv chain and output | blocks next Talker step and sink callback |
| optional decoder thread | decoder consumer plus same engine pool | queued complete decoder calls | can decouple caller boundary, but same-pool submission and resource contention remain |
| PCM output | callback caller, synchronous writes | float-to-PCM and three writes | slow client can hold inline engine or decoder consumer |

Important nuance: the actual normal loop order is admission, Talker/codec-head
and CP for the current iteration, decoder delivery if its target is reached,
optional M1 admission, batched decoder delivery if enabled, then the next
Talker step. It is sequential and decoder-blocking, but it is not literally a
fixed Talker -> CP -> decoder -> Talker sequence in every iteration.

The measured utilization therefore identifies an under-filled/coupled serving
regime, not an already-proven independent compute budget. Before using idle
cores for overlap, AR-2 must separate scheduler serial time, pool submit/wait,
memory stalls, and useful kernel time.

## 4. Prefill resumability verdict

The current model/runtime state supports completed-state continuation, not
arbitrary prefill continuation.

Surviving state after a completed Talker prefill:

- Talker KV cache and kv_len;
- final hidden/decoder input used to start generation;
- the full input embedding copy used for prefix comparison in the normal
  generation context;
- allocated prefill buffers that are reused by the context.

Not present as a resumable request-owned state:

- a prefill token cursor;
- per-layer partially completed attention/FFN state;
- a safe return point between layers/tokens with all required scratch;
- a server admission object that owns such a cursor.

The server then copies completed K/V into the per-slot batch state and starts
frame generation. Decoder state has its own K/V, latent, VQ pad, and causal
tail/carry fields, but those are post-generation streaming state and do not
make prefill resumable.

The AR-1 prefill direction is usable only if AR-2 explicitly scopes it to a new
fixed-sequence incremental execution state machine. It must not claim that the
existing prefix-cache path already provides this.

## 5. Decoder slice/preemption verdict

The current decoder has a useful completed-call streaming state boundary:
qwen_sd_stream_state_t persists transformer caches, latent history, VQ padding,
and convolution tails/carries across calls. That supports frequent calls with
correct causal state.

It does not currently have a resumable intra-call boundary. In particular:

- ConvT carries and convolution tails are updated as part of a completed stage;
- ragged offsets and total columns are local to the batch invocation;
- the signal pointer moves through newly allocated intermediate buffers;
- final per-request audio allocation/scatter happens after the whole chain.

AR-2 should therefore distinguish:

1. smaller complete calls, already available through q8/q4 target selection;
2. a decoder job that can yield only at completed stages;
3. a true strip executor with persistent stage/cursor state.

Only item 3 deserves the literal name resumable/preemptible decoder. A first
implementation may use item 1 or a completed-stage boundary, but its claims and
kill criteria must say which one it implements.

## 6. Same-pool overlap constraints

The shared engine pool has one global job slot and a submit mutex. A caller
executes its own chunks while pool workers execute the rest, then waits. A
second caller can exist, but it waits to publish a job while the first
submission holds the mutex. This permits overlap in caller-side serial sections
only when the timing and ownership line up; it is not two independent pool
teams.

Safe assumptions:

- no nested pool team should be created;
- decoder consumer and engine loop must have explicit queue ownership and
  completion rules;
- output callback must not accidentally reintroduce a blocking engine
  dependency;
- pool width, caller threads, CPU affinity, and BLAS thread ownership must be
  recorded together;
- per-stream decoder states must never be concurrently submitted for the same
  stream.

Unknowns that remain genuinely open:

- whether decoder serial glue overlaps useful Talker/CP caller work in the
  actual timing regime;
- whether the pool submit mutex turns overlap into queueing;
- whether the resulting shared weight/cache and DRAM traffic reduces or
  increases sustained RTF;
- whether stream cadence improves once output callbacks move away from the
  engine loop.

## 7. Talker/CP batching feasibility

Current data layout is more favorable than a greenfield design:

- Talker K/V and CP K/V are already slot-indexed;
- active slots are compacted for batched regions;
- Talker positions are passed per slot;
- CP active masks preserve slot ownership while batching each codebook stage;
- output/RNG/sampling remain request-local.

The missing representation is not a second copy of every state tensor. It is a
global scheduling/ownership layer that can select ready slots across the current
prefork boundary without violating per-slot positions and CP codebook order.

The theoretical benefit is weight-read amortization. The measured B2/B1 ratio
supports that opportunity. Implementation feasibility and B4/C4 scaling remain
unresolved until a single-owner prototype isolates:

- ready-set construction;
- admission and cancellation;
- per-slot KV/state access;
- same-codebook CP batching;
- output routing and slow-client behavior;
- decoder work being bounded or decoupled.

This is sufficient to keep EO-2 alive as a later, gated direction, not sufficient
to claim it is the next guaranteed win.

## 8. P2 retrospective corrections

| AR-1 classification | Audit verdict |
|---|---|
| Direct ConvT | VERIFIED BY CODE/EVIDENCE as a byte-identical/local optimization result. OVERSTATED if generalized to all shapes or all end-to-end regimes; it removed a copy/accumulation cost without changing the ragged call structure. |
| Direct depthwise | VERIFIED as implemented and byte-identical in the recorded gates. The neutral/slight result does not prove depthwise is universally irrelevant; it says this realization did not move the serving envelope. |
| Direct input / warm range | VERIFIED as a per-slot warm-path optimization and correctly recorded as not active in the ragged two-slot implementation. “Effectively irrelevant to C4” needs the qualifier that singleton/per-slot calls can still use it; the call-mix denominator matters. |
| Fused residual AMX | VERIFIED as a real mechanism in both per-slot and ragged code, default-off, with a numerical/audio caveat. OVERSTATED when described as a proven serving improvement or as exactly 12 fewer “rendezvous”; it changes the 1x1 residual route and needs path-specific counters. |
| BLAS-C residual | The rejection result is supported for the tested realization. The explanation about OpenBLAS beta=1 packing/partition behavior is PLAUSIBLE BUT REQUIRES EXPERIMENT unless backed by a direct oracle; do not promote it to a universal BLAS rule. |
| Direct gather/quant / one-row quant | VERIFIED that the tested one-row realization was byte-identical locally but regressed C4 and was reverted. OVERSTATED if this rejects panel-wide direct preparation as a concept; the evidence rejects the granularity, not all direct preparation. |
| Low-N split | VERIFIED as rejected for the recorded M=768/C4 workload: more tasks and shorter tiles regressed. OVERSTATED as a universal theorem that no future row decomposition can help under a different state/aggregation model. |
| Cross-request decoder aggregation | VERIFIED that existing ragged batching concatenates compatible active items into one logical workset and that the 1x12/batch4 probe formed multi-item groups. The poor probe is valid negative evidence for that coupled configuration, not proof that all aggregation is useless. It also does not create a workset across prefork workers. |
| Warm output-range/direct INT8 | VERIFIED as scope-limited to the per-slot warm path; the ragged two-slot path uses rag_conv1d and does not receive that optimization. The “no-op for ragged” statement is correct; the stronger whole-C4 claim needs call-mix qualification. |
| Ragged panel parallelization | VERIFIED as the important P2 serving improvement. It parallelizes panels within an existing call; it does not solve caller-level burst/cadence coupling. |

The P2 evidence supports a structural next step, but not the stronger statement
that every byte-level optimization is irrelevant. Several changes can still
matter after the critical path is changed; their previous C4 results only bound
their value in the tested architecture.

## 9. Fused residual implementation verdict

Keep the fused residual path as a later qualification candidate and as a
possible building block inside a future strip body. Do not drop it, but do not
make it the default or use its short positive signal as a serving conclusion.

The code-level mechanism is coherent:

- rag_conv1d_fused_residual is flag-gated;
- it obtains the existing persistent Design-D weight pack;
- the Design-D residual epilogue writes projection plus residual into a separate
  result buffer;
- the caller retains the input until the AMX call returns, then replaces the
  signal and releases the old buffers
  (qwen_tts_speech_decoder.c:3326-3344, 3661-3692).

Minimum promotion gate, to be run by a later implementation owner:

- serial/per-slot and ragged deterministic golden comparisons on the same
  inputs, including partial lengths and causal tails;
- byte identity where the arithmetic/order is intentionally unchanged;
- otherwise report max/mean error and per-stage relative RMS, then pass the
  established mel/audio correlation and duration/artifact gate;
- require no cross-request state contamination, no stale/skipped output, and
  exact bias/tail behavior;
- only after numerical gates, run a server A/B with zero errors, rejects,
  timeouts, underruns/starvation events, and no material playback regression.

The Fable thresholds mel_corr >= .99, duration delta <=5%, relative RMS around
1e-2, and no audible artifacts are reasonable provisional gates, but they are
not evidence that the fused path has already passed them.

## 10. Architecture assumptions Fable must weaken or change

1. Replace “not decoder arithmetic/AMX reach” with “serialization/coupling is
   the leading measured serving hypothesis; whole-request AMX wall share and
   non-AMX decoder cost remain unknown/material.”
2. Rename “102 pool rendezvous” to “102 source-level operation/call-site
   inventory for a nominal ragged q8 chain”; report actual pool submissions and
   waits separately.
3. Remove the claim that prefix-cache pos0 mechanically enables resumable
   prefill. Define a new cursor/state machine as an implementation requirement.
4. Do not call q2/q4 intra-decoder preemption. They are smaller complete calls
   unless persistent stage state is added.
5. Remove “recover the idle third” as a prediction. Shared-pool overlap needs
   evidence, and pool width is not automatically reduced for a consumer thread.
6. State that current ragged batching already aggregates compatible active items
   within one decoder call; the missing axis is cross-worker ownership and/or
   cadence decoupling, not batching in the abstract.
7. Scope the warm strip result to its per-slot path and qualify its relevance by
   the C4 singleton/ragged call mix.
8. Keep blocking socket writes as a latent coupling risk, not as a measured
   dominant term in the current KPI evidence.
9. Keep static lane rejection conditional on the current C<=4 evidence and
   utilization model; do not treat it as a permanent hardware conclusion.
10. Keep appending text after generation and arbitrary text segmentation out of
    implementation scope until sequence layout, training semantics, and state
    equivalence are established.

## 11. AR-2 disposition of major proposed moves

| proposed move | disposition | required correction |
|---|---|---|
| Cooperative bounded-quantum engine loop | ACCEPT WITH CORRECTIONS | Accept the direction for cadence, but define the first boundary honestly: smaller complete decoder calls or completed-stage slices are not full preemption. Do not promise capacity gain from serialization removal alone. |
| Full/incremental fixed-prompt prefill | ACCEPT WITH CORRECTIONS | Treat as a new request-owned prefill state machine. Prefix cache is a reuse input, not proof of pause/resume. |
| Appending new text after codec generation | REJECT | This is a model sequence/layout change, not a mechanically safe scheduler feature. |
| Lead-aware scheduler | ACCEPT WITH CORRECTIONS | Keep lead/deadline state separate from decoder preemption. First validate decisions at existing safe call boundaries. |
| Decoder strip/full SQ-1 body | ACCEPT WITH CORRECTIONS | Use it inside an explicit bounded-state design and preserve ragged/per-slot ownership. Do not assume the per-slot warm path solves ragged C4. |
| Same-pool decoder overlap | ACCEPT WITH CORRECTIONS | Gate behind a small A/B; no automatic worker reservation, no “idle third” claim, and require playback plus resource evidence. |
| Output isolation/nonblocking writer | ACCEPT WITH CORRECTIONS | The current writes are synchronous. Define bounded queue/backpressure, cancellation, memory limits, and exact client-disconnect semantics before calling it safe. |
| Fused residual AMX | ACCEPT WITH CORRECTIONS | Retain default-off as a candidate; require numerical/audio and server gates before promotion. |
| Generic cross-request decoder-width aggregation as the immediate fix | REJECT | Within-worker ragged aggregation already exists and the negative probe did not isolate cadence. Revisit only with a concrete ownership/cadence hypothesis and exact scatter mapping. |
| Single-engine/global Talker/CP batching | ACCEPT WITH CORRECTIONS | Keep as later EO-2 work after decoder/output coupling is bounded. Existing per-worker B>1 is evidence of opportunity, not proof of global feasibility. |
| Static Talker/decoder core lanes at current C<=4 | REJECT | Current evidence does not justify partitioning; revisit only after a measured compute-bound regime or a bounded overlap experiment. |
| BF16/W4/VNNI/ARM expansion in this AR-2 step | REJECT | Outside this implementation audit and explicitly parked by the frozen P2 scope. |

## 12. Final audit recommendation

AR-1 has a sound high-level serving diagnosis and a useful dependency ordering,
but it must be consumed as a hypothesis document with the corrections above.
The most important hard corrections are prefill state, literal preemption,
pool-rendezvous accounting, and the absence of proof for same-pool overlap or
global batching.

AR-1 SAFE TO USE FOR AR-2: YES WITH CORRECTIONS
