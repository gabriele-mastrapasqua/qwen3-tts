# Post-P4 evidence synthesis and P5 architecture decision — 2026-09-08

Task: AR-1 (second-pass synthesis after P3/P4; analysis only, no runtime change and no
P5 implementation). The Codex audit and bounded F1 falsifier are appended below; F1 was
run as a flags-only C4 screen and PLAN/evidence are updated with that result. Section 13
remains a proposal, not current runtime state.

## Question

Given every PASS/FAIL result through P4, what is the smallest coherent serving
architecture that improves the C4 playback envelope and could make a fifth or sixth
concurrent request interactive without destroying batching, locality or steady-state
throughput — and which experiments freeze that design?

## Known facts (ground truth at HEAD)

- The synthesis was audited from HEAD `e60aebf65cd8d068c7ff89c2a5a0cd8e5511cd6a`,
  branch `feature/x86-amx-vnni-oss`, with no runtime diff. The synthesis itself was
  the only untracked input at audit start; the earlier “three untracked AR-1 notes”
  description was stale. Runtime-relevant history includes SL-1, output isolation,
  the lead-gate experiment, MT-4, the scratch-reuse revert, claim-first, the fused
  residual counter, and parent fail-fast (`3db3b76`), plus their evidence notes.
- Reference serving configuration used by every P3/P4 note: 2x6 prefork, batch cap 2,
  q8, ragged threshold 2, Design-D INT8 AMX, warm strip, engine pool, synchronous
  output. **No committed profile encodes it**: `configs/perf/gcp-c4-standard-24-vnni-ttfa.json`
  is the VNNI-only, `status: unqualified` file with `batch_size 8`, `max_queue 1` and
  `QWEN_DECODER_BATCH=0`. Code defaults differ from the reference in two places:
  `QWEN_SD_RAG_MIN_PANELS` defaults to 8 (reference runs use 2) and `QWEN_SD_AMX_D`
  defaults off (reference sets it). This is a control-plane gap, recorded below.
- Flag defaults at HEAD (all default-off unless stated): `QWEN_TTS_STREAM_LAYOUT`,
  `QWEN_SERVER_ASYNC_OUTPUT` (1 MiB byte bound, 5 s send timeout, overflow fails the
  stream), `QWEN_STREAM_LEAD_GATE` (target 250 ms), `QWEN_SD_FUSED_RESIDUAL`,
  `QWEN_DECODER_THREAD`, `QWEN_PREFILL_HELPER`, `QWEN_ADMIT_M1`, `QWEN_TTFA_PRIORITY`,
  `QWEN_SD_STREAM_STRIP`, `QWEN_SD_DIRECT_*`; `QWEN_DECODER_BATCH` is set to 1 by the
  server; `QWEN_STREAM_DECODE_CHUNK` 8; gang lead/min 4/2; `--max-queue` unset means
  child grace cap 1 and parent listener gating (see admission).
- Serving loop at HEAD (`qwen_tts.c:2629`): admission scan with inline full prefill
  (`ADMIT_PREFILL` at 2988) → BLAS budget → lead gate (3249-3267; when no slot is
  eligible the loop sleeps 1 ms and skips frame, decode and admission) → narrowing
  masks → codec head → sampling → CP → frame record + trailing-text or `tts_pad`
  embedding (3081-3086) → decode delivery (ramp 1, 2, 4 then chunk at 3338-3344; no
  floor in code) → optional M1 admission → inline ragged gang (3457-3507) → Talker step
  (3526). `QWEN_DECODER_THREAD=1` forces `dec_batch = 0` (2818).
- Admission at HEAD (`qwen_tts_server.c`): `listen(fd, 16)` (1444); the parent polls
  the listener only when `free_slots > 0 || reject_full_at_parent` (2719);
  `reject_full_at_parent = (g_cfg_max_queue == 0 && !QWEN_QUEUE_UNBOUNDED)` (2575);
  full workers → `503 all workers at capacity` (2792-2797). Default keeps the old
  behavior: the connection stays un-accepted in the kernel backlog. Per child, the
  parent cap (2) binds before the child grace slot (`slots + cap = 3`), so the child
  grace queue and `--queue-timeout-ms` are structurally unreachable under prefork.
  Slot release is whole-connection (`srv_conn_close` after `sink_on_done`). The batched
  header is written at admission in `sink_next_job` (1822-1841) on both sync and async
  paths; `TCP_NODELAY` is set in `set_client_timeout` (1373-1379).
- Decoder at HEAD: fused residual gates on `kernel == 1`, `in_ch == out_ch`, Design-D
  and a cached pack; per-slot fallback is `causal_conv1d`, ragged fallback is
  `rag_conv1d` plus an explicit add, so the ragged route moves from FP32 SGEMM to INT8
  Design-D with the residual in the epilogue. Claim-first allocation survives the
  scratch revert; cross-call reuse is gone. Two active items with Design-D take the
  ragged path; one item takes the per-slot path where the warm strip applies.
- Doc/code disagreements: PLAN "blocking writes, no send timeout" holds only for the
  default synchronous path; PLAN "~41 rendezvous" and the historical "~102" are
  source-level call-site inventories, not measured pool barriers, and no per-chunk
  pool-submission counter exists; PLAN "SL-1 pending ICL/clone" understates that the
  ICL branch is implemented (`qwen_tts.c:1342-1390`) and only its gate is missing
  (`tests/stream_layout_smoke.sh` asserts the non-ICL value).

## Unknowns

- Per-worker Talker step wall at B=3 and CP wall at B=2/3 (only B1/B2 Talker measured).
- Pool submissions and pool wait per decoder call (no counters).
- Quality of the streaming layout for presets and ICL against the official runtime.
- Whether the fused-residual gain holds on other voices/models (tested contract only).
- Natural cohort coincidence across workers at C4/C5 (never traced).

## Codex adversarial audit — 2026-09-08

This audit checks the synthesis against the e60aebf source and the accumulated evidence.
The classifications below apply to the claim as written, not to the underlying
experiment when the claim is broader than that experiment.

### Exact runtime defaults at the audit HEAD

| setting | current code/default | qualification consequence |
|---|---|---|
| `QWEN_TTS_STREAM_LAYOUT` | off | current control is the non-streaming layout |
| `QWEN_SERVER_ASYNC_OUTPUT` | off; 1 MiB / 5 s apply only when enabled | synchronous writes remain on the reference path |
| `QWEN_STREAM_LEAD_GATE` | off; target defaults to 250 ms when enabled | the positive fused/quantum evidence is not a lead-gate result |
| `QWEN_SD_FUSED_RESIDUAL`, `QWEN_SD_AMX_D`, `QWEN_SD_STREAM_STRIP`, `QWEN_SD_DIRECT_*` | off | fused residual, warm strip and direct-copy paths are explicit experiment arms |
| `QWEN_DECODER_BATCH` | server sets `1` unless explicitly disabled | decoder batching is part of the serving control |
| `QWEN_STREAM_DECODE_CHUNK` / busy override | 8 / off | the loop ramps 1, 2, 4 before the configured steady-state quantum |
| decoder gang lead / minimum | 4 / 2 | these are collection controls, not cross-request matrix fusion |
| `QWEN_SD_RAG_MIN_PANELS` | 8 | prior reference runs explicitly set 2; this control-plane difference matters |
| `QWEN_DECODER_THREAD`, `QWEN_PREFILL_HELPER`, `QWEN_ADMIT_M1`, `QWEN_TTFA_PRIORITY` | off | rejected or diagnostic-only variants are not current behavior |
| prefork `--max-queue` | unset: parent listener gating plus child grace cap 1 | `--max-queue 0` is an explicit fail-fast arm, not the default |

### Major-claim classification

| synthesis claim | classification | implementation correction |
|---|---|---|
| The dominant C4 limitation is engine-loop serialization/coupling rather than decoder arithmetic. | SUPPORTED BY CURRENT CODE + SUPPORTED BY MEASURED EVIDENCE, scope-limited | The loop really owns admission, Talker/CP, inline decoder and synchronous delivery; fused and panel-parallel changes improve the critical path. This does not prove decoder arithmetic is cheap, that all idle cores are usable, or that memory bandwidth leaves free capacity. |
| C5/C6 are primarily a fixed-slot/admission problem. | SUPPORTED BY MEASURED EVIDENCE, with a leading-cause inference | Parent/child timestamps and the cap-0 selection arm support pre-admission waiting. Exact backlog share and the cost of a third slot remain UNKNOWN; do not state that cap 3 is affordable. |
| The additive per-iteration model and its q/fused savings are empirical measurements. | OVERSTATED | Decoder intercept/slope are fitted from phase traces; Talker/CP terms and addition are modeled estimates. The ~3.4 ms and ~15% values are derived conversions, not direct per-iteration decompositions. |
| Fused residual shifts the q2/q4/q8 frontier. | UNKNOWN / NEEDS TEST as a causal interaction; SUPPORTED BY MEASURED EVIDENCE for the joint arm | F1 establishes fused-on q2/q4/q8 behavior and makes q4 the next C4 candidate, but no exact fused-off paired arm isolates a shift; q8 remains the throughput control. |
| The historical ~102 decoder rendezvous is a measured per-chunk count. | MECHANICALLY INACCURATE | It is a source-level call-site inventory. Engine calls, panel dispatches, snake calls, BLAS work and actual pool submissions/barriers must not be collapsed; actual pool wait is UNKNOWN. |
| SL-1 has flat prefill semantics in production. | SUPPORTED BY CURRENT CODE for the known-text branch; SUPPORTED BY MEASURED EVIDENCE for its tested non-ICL arms | The branch and ICL overlay exist, but preset/ICL quality and long-input server behavior are not closed. “~73 ms flat” is a measured subset, not a universal runtime guarantee. |
| Prefork slot/grace behavior makes the default child queue the C5 mechanism. | SUPPORTED BY CURRENT CODE | The parent gates acceptance while full by default; its cap-0 behavior is different. A request can be delayed before child `t_recv`, so child queue timeout cannot account for that delay. |
| Warm strip is irrelevant to ragged two-slot C4. | SUPPORTED BY MEASURED EVIDENCE, scope-limited | In the tested two-item ragged route the strip is not the active path; this does not generalize to C1/per-slot work or to a future aggregated workset. |
| Fused residual is ready for unconditional deployment. | STRONG INFERENCE, quality-gated | It has strong C4 evidence and remains a later qualification candidate, but the quality bank and cross-voice/model contract are incomplete; the flag is still off. |
| B2/B1 proves a single global Talker/CP engine is the next capacity move. | WEAK INFERENCE | The ratio exposes a bandwidth opportunity in the measured context. It does not establish cheap state gathering, CP progress compatibility, or a no-wait cohort across prefork workers. |
| Cap=3 is the next bounded falsifier and utilization-aware admission is the architecture. | STRONG INFERENCE / UNKNOWN capability | Cap 3 is a justified next test because C5 selection is clear, but the utilization rule and its threshold are unimplemented and unmeasured. |
| The section-13 architecture is the current runtime. | INCORRECT if read as present state | It is a proposed post-F1/F2 design. Current defaults still have fused/D/strip/stream layout/async output/lead gate off, and the parent does not yet make a measured `U_max` decision. |

The audit therefore supports the synthesis as a useful evidence/inference document,
but not as proof that decoder arithmetic, memory pressure, or a third slot are already
solved. F1 is the only new measurement authorized by this checkpoint.

## Files/functions inspected

`qwen_tts.c` (`qwen_tts_serve_continuous`, `qwen_tts_generate` layout branches,
`RECORD_FRAME_AND_EMBED`, `dec_worker_main`, helper), `qwen_tts_server.c`
(`qwen_tts_serve_prefork`, `sink_next_job`, `sink_on_chunk`, `sink_step_allowed`,
`stream_output_*`, `set_client_timeout`, `jq_push`), `qwen_tts_speech_decoder.c`
(fused residual gates, `sd_rag_panel_worker`, batch dispatch), `qwen_tts.h`
(`qwen_sd_stream_state_t`), `configs/perf/*.json`, `tests/serve_parallel_wave.py`,
and every `.work` note listed in section 2.

---

## 1. Ground truth summary

The server is the same lockstep engine loop per prefork worker that AR-1 described,
with four things added: an optional known-text streaming layout, an optional detached
output writer, an early header plus `TCP_NODELAY`, and an optional parent-side
fail-fast. Everything else that was tried (lead gate, prefill helper, decoder consumer,
scratch reuse) is default-off or reverted. The only serving-positive runtime change
since P2 is the fused Design-D residual epilogue, default-off.

## 2. Evidence table

Bank note: quantum-floor and lead-gate runs used `load_texts_en.txt`; fused residual,
consumer, scratch and claim-first runs used the short diverse bank; only paired arms
within one note are compared below. All runs: 2x6, cap 2, Design-D, threshold 2.

| mechanism | hypothesis | implementation | scope | result (paired) | confidence | proves | does NOT prove |
|---|---|---|---|---|---|---|---|
| SL-1 known-text layout | prefill stops scaling with text length | `QWEN_TTS_STREAM_LAYOUT`, prefill ends at `text[0] ⊕ bos`, trailing text per step | CLI + server, non-ICL gated, ICL implemented ungated | prefill positions 15/33/45 → 10; ms 85/212/228 → 73 (C1 CLI, 4/22/34 tokens); C1/C2 server TTFA 99→78, 105→82 ms | high (mechanism) | the text-length prefill term is removed for known text | quality parity (model-visible change), ICL/clone parity, long-input server envelope |
| Output isolation | slow clients must not block inference | per-stream bounded queue + writer thread | sync vs async at C1-C4 | KPI within noise (C4 p95 0.821→0.830), slow-reader isolation proven, byte-identical audio | high | correctness/isolation; transport off the engine thread | any inference speedup; thread/memory cost at long concurrency |
| MT-4 | TTFB is a separate event | header at admission, `TCP_NODELAY` | batched path | C3 TTFB p95 61 ms vs TTFA 167 ms; C5 TTFB p95 4.28 s | high | metric truth; C5 tail is pre-admission | any speedup |
| Quantum floor q1/q2/q4/q8 | cadence vs fixed-cost frontier | existing knob, one wave C3/C4 | control | C4 STREAM p95 1.005/0.892/0.862/0.817; prebuffer p95 224/149/277/333 ms; stall@250 0/0/0/50 % | medium (one wave) | a real frontier; q1 unaffordable; q2 strongest cadence | which quantum is optimal after fused residual |
| Hard lead gate | parking above 250 ms lead improves cadence | `QWEN_STREAM_LEAD_GATE`, loop sleeps when none eligible | C3/C4 | 95.8 % of checks suppressed; cores 7.3→5.2; STREAM p95 0.838→0.986; stall unchanged | high | this realization removes useful work and gains nothing | that lead is useless as an ordering or sizing signal |
| Prefill helper + LOW | moving prefill off the loop protects streams | cloned-context one-shot helper, LOW priority | C3/C4 | TTFA p95 435→2379 (C3), 503→2459 ms (C4); STREAM p95 C4 0.821→0.883 | high | one-shot async prefill starves on a busy pool | anything about a bounded resumable prefill |
| Same-pool decoder consumer | overlap decoder with Talker on one pool | `QWEN_DECODER_THREAD`, forces `dec_batch=0` | C3/C4 | STREAM p95 0.847→1.296; TTFA p95 174→1126 ms; max-gap 511→1286; csw ×2.9; cores flat; group=1 | high | this realization serializes on the pool and loses ragged batching | that all overlap is impossible |
| Ragged scratch reuse | allocation is a material fixed cost | per-pthread retained buffers | C3/C4 | C4 p95 0.810→0.843, prebuffer 344→420 (reverted) | medium | allocation churn is not the dominant term | that allocator cost is zero |
| Claim-first | avoid allocating for workers that claim nothing | allocate after first claim | C3/C4 | KPI-neutral; 12 % fewer allocation events (derived) | high | hygiene only | any serving effect |
| Fused residual Design-D | remove the 1x1 residual-add pass | INT8 epilogue, ragged route FP32→INT8 | C3/C4 A/B, 5-min C4 SOAK, C5/C6 screen | C4 STREAM p95 0.831→0.788; prebuffer 389→266; safe-start 568→426; stall@250 17→0 %; SOAK p50/p95 0.830/0.893, TTFA p95 526, safe-start p95 917, stall@500 1/119 | high (C4) | shortening the inline decode call improves throughput AND cadence together | decoder intercept solved; quality on every voice; per-class drift |
| C5/C6 screen | capacity beyond C4 | same binary, cap 2 | 3 waves | STREAM p95 0.852/0.802 but TTFB p95 4.28 s, TTFA p95 4.44/4.77 s, safe-start p95 4.6/5.0 s; B 2.38/2.44 | high | failure is pre-admission slot wait, not compute | that a 5th slot would be affordable |
| Parent fail-fast | overload must be visible | `--max-queue 0` → 503 at parent | C5 two waves | 8 accepted, 2 rejected; accepted TTFB p95 187 ms, STREAM p95 0.788 | high | honest overload semantics | capacity increase (selection effect on accepted set) |
| Ragged threshold 2 (pre-P2) | let small-panel calls use the pool | knob | C4 | p95 1.0035→0.954 | medium | pool participation helps | cadence |
| Ragged panel parallelization (P2) | parallel panels in a call | atomic cursor | C4 | p95 1.338→0.870 | high | the one large structural win | |
| F1 fused residual × quantum | fused residual may make smaller complete decoder calls viable | flags only; fused on, q2/q4/q8 | C4, 3 waves, short diverse bank | q4 STREAM p95 0.868, prebuffer p95 201 ms, stall@250 0%; q2 0.914/276 ms/0%; q8 0.822/306 ms/25% | high for this joint arm | q4 is the next C4 playback candidate; q8 has higher throughput | no causal fused-vs-off delta; no 5-minute qualification |

## 3. Causal model

Boundary by boundary, with the resource, the blocking operation and the evidence:

| boundary | owner / serialization | queue | blocking op | cohort | persistent state | resource | latency term | evidence |
|---|---|---|---|---|---|---|---|---|
| arrival → listener | kernel; parent polls only with a free slot (default) | kernel backlog 16 | none accepted; wait = shortest occupying request | — | — | socket | 0 at C≤4; ~4.3 s at C5/C6 | C5 TTFB p95 4.28 s; timeline reconstruction |
| parent admission | parent, least-loaded routing, cap = batch size | none (grace slot unreachable) | fd pass | — | `active[w]` | worker slot | ms | fail-fast note |
| worker ownership | one engine loop per worker | `jq` (count ≈ 0) | — | slots in one process only | KV, CP KV, decoder state per slot | pool, LLC, DRAM | — | audit 2.6 |
| prefill | engine thread, inline, one-shot | — | full prefill 46-394 ms (non-streaming); 73 ms flat (SL-1) | none | KV into slot | pool, DRAM (bf16) | first-play; stalls neighbors | input-length note; SL-1 note; helper note |
| Talker step | engine thread + pool region, lockstep over B_eff | — | region ~22 ms (B1), 24 ms (B2) | slots in worker; B_eff 1.8-2.4 at C4 | KV | DRAM-bandwidth | continuity, throughput | CT-4 |
| CP | engine + region, 16 sequential codebooks | — | ~12-16 ms | slots | CP KV | L3 bandwidth, barriers | continuity | CP audit |
| decoder | engine thread, inline ragged gang | frames pending per slot | whole call: 25 ms + 9.5 ms/frame/item | ready items in worker | tails/carries | pool, cores | burst → cadence; per-frame cost → throughput | CT-2; quantum floor; fused |
| PCM output | engine thread (sync) or writer (async) | per-stream queue (async) | blocking write (sync) | — | — | socket | latent | OUT note |
| network → client | client | — | — | — | — | — | client-observed marks | MT-1 |

Modeled per-iteration model at C4 (per worker, B≈2, 80 ms frame period), constructed
from paired diagnostics rather than directly measured as one additive decomposition:

`iteration ≈ Talker(B) + CP(B) + decoder_amortized(q, items) + serial`

`decoder_amortized(q, 2) = (25 + 2·q·9.5)/q` → q8: 22 ms, q4: 25 ms, q2: 31.5 ms, q1: 44 ms.
With Talker 24 and CP ≈ 16 (estimate): q8 → 62 ms (ρ ≈ 0.78; near measured p50
0.79-0.80), q2 → 71 ms (0.89; near measured C4 p95 0.892), q1 → 84 ms (1.05; near
measured 1.005). This is a useful strong inference, not a fitted or causal proof: the
decoder slope/intercept came from `[SDPHASE]` fits, while the Talker/CP terms and the
additive decomposition are estimates. The derived `~3.4 ms/iteration` and `~15 %`
fused saving are conversions from the q8 A/B RTF difference and this model, not direct
per-iteration measurements. The twelve residual FP32 SGEMM entries are a source-level
inventory; they are not twelve measured pool rendezvous.

What the model says about the opposite results:

- Anything that lengthens the inline decode burst (larger q, aggregation) raises the
  chunk gap `q·iteration + burst` and the required prebuffer, even while RTF improves.
- Anything that shortens the burst on the critical path (fused residual, ragged panel
  parallelization, threshold 2) improves cadence and throughput together.
- Anything that adds a second submitter to the same pool (decoder consumer, helper)
  pays `submit_mtx` serialization plus lost cohorts: the consumer's per-frame call
  waited behind Talker regions and reached 1.4 s; the helper's LOW prefill waited for
  idle windows that a C3/C4 pool rarely has.
- Anything that parks Talker work (lead gate) leaves the loop with nothing else to do:
  the loop sleeps 1 ms and utilization falls. In a two-slot lockstep worker there is no
  competing work to give the freed time to, and parking one stream drops the other to
  B=1 (per-slot cost +20 %). The gate therefore converted a 0.83 server into a 0.99 one
  by construction: production rate was clamped to playback rate.
- Allocation is not a material term: reuse and claim-first moved nothing.
- Above C4 the binding term is not in the iteration at all: with four whole-connection
  slots, a fifth request waits for the shortest occupying request (~4.3 s, independent
  of C), before any child timestamp exists.

## 4. Cross-experiment hypotheses

**H1 — locality/cohort preservation matters more than fine-grained concurrency.**
SUPPORTED. The three concurrency-adding realizations (consumer, helper, gate-induced
narrowing) all lost; the three critical-path shortenings (ragged parallelization,
threshold 2, fused residual) all won, and the consumer's loss coincided exactly with
losing the ragged gang. Not coincidental: the model in section 3 predicts each sign.

**H2 — reducing work on the existing critical path is more reliable than moving work to
other threads.** SUPPORTED with one qualification: it holds when the removed work is
material (fused residual, ragged panels) and is neutral when it is not (scratch,
claim-first, direct copies in P2). SL-1 supports it on the prefill term; OUT is neutral
because socket writes on loopback were never material. No counter-example exists.

**H3 — C4 and C5 have different limiting mechanisms.** SUPPORTED. C4 sits on the
iteration/cadence frontier (STREAM 0.79-0.89, prebuffer 266-596 ms). C5/C6 keep STREAM
0.80-0.85 and B ≈ 2.4 while TTFB p95 jumps to 4.28 s, identical at C5 and C6; the wait
is slot occupancy before admission. This is a phase change in the dominant term, not a
monotonic compute ceiling.

**H4 — preserving batching beats maximizing instantaneous core occupancy.** SUPPORTED
for this loop: B2 is 0.83× B1 per slot (CT-4); the gate and the consumer both reduced or
destroyed cohorts and lost; cores were 7-9 in winning arms and did not rise in losing
arms (consumer 8.8-9.0 with ×2.9 context switches). Caveat from the audit: measured
core-equivalents do not prove exploitable idle capacity, so H4 is about not losing
cohorts, not about filling cores.

**H5 — overload policy and compute scheduling should be separate mechanisms.**
SUPPORTED by structure: fail-fast changed overload semantics with zero effect on the
iteration; the gate changed the iteration with zero effect on overload. The five
mechanisms (admission/fail-fast, startup priority, steady-state cadence, cohort
formation, decoder quantum) act on different boundaries in section 3 and should stay
independent. No evidence calls for one scheduler.

## 5. The central question

### A. Best C4 architecture

| candidate | disposition | reason |
|---|---|---|
| fused residual | deployment-selected now; production default after the quality bank and a per-class-sampled SOAK | only serving-positive P4 change; SOAK hard gate passed in all windows |
| decoder quantum | q4 is the next C4 reference candidate; q8 remains the throughput control and q2 is mandatory-realtime but misses the preferred p95 | F1 jointly supports fused+q4 for playback; it does not isolate a fused-vs-off shift |
| async output | deployment-selected; default after a 5-min C4 thread/memory qualification | correctness; KPI-neutral; removes coalescing |
| MT-4 | production (already default) | metric truth |
| parent fail-fast | deployment-selected for interactive SLAs (`--max-queue 0`); default unchanged | honest overload |
| SL-1 | deployment-selected per mode after F5 (clone first, where streaming is the upstream default; presets after quality) | removes the text-length term; model-visible |
| ragged threshold 2, Design-D, warm strip | production reference; must be written into a committed profile | control-plane gap |
| lead gate, helper, decoder consumer, scratch reuse | diagnostic-only / reverted | rejected realizations |
| claim-first | retained hygiene | neutral |

### B. Path toward interactive C5/C6

Ranked by compatibility with the evidence:

1. **Admission-aware slot accounting** (leading mechanism, capability UNKNOWN):
   the fifth request waits for a slot before child service in the tested C5/C6 arms.
   A third slot on one worker would add Talker(B3) + CP(B3) + decoder for three items;
   the B1→B2 ratio 1.10 and the decoder slope (+76 ms per 8-frame call per item) make
   a B=3 worker cost model plausible, but the ρ ≈ 0.9-1.0 estimate at q8 is modeled,
   not measured. F-cap3 is therefore the first bounded capability falsifier. Only if
   it passes should utilization-aware admission be designed around a measured margin.
2. **Per-stream cost reduction** (SUPPORTED as the lever that makes a third slot
   affordable): fused residual (done), quantum policy (F1), intercept/rendezvous work.
3. **SL-1** (SUPPORTED for the tested startup mechanism): a fifth request's own first
   play after admission is ~425 ms p50 on the mixed bank; with SL-1 the measured
   non-ICL prefill part falls to ~73 ms in the tested bank. Preset/ICL quality and the
   established-stream effect at long input remain open, so the inline hole cannot yet
   be claimed to shrink in every mode.
4. **Global Talker/CP cohorts** (WEAK INFERENCE for C5 on this host): per-stream Talker
   cost would fall, but the inline decoder cost for five items on one thread would rise
   to ~62 ms per frame; without decoder off the critical path (rejected realization)
   the single engine reproduces the 1x12 probe. Does not address the slot wait.
5. **Controlled overlap** (UNKNOWN): only with a design that keeps the ragged gang and
   submits chunky work; no such design exists.
6. **Different topology** (WEAK INFERENCE): 3x4 was worse in the old sweep; 1x12 needs
   4-5; a 12-core host has no better static split.

### C. Global Talker/CP batching as P5

After P3/P4 the current recommendation is: **not on this host, not as the next step.** The B2/B1
amortization is real, but on 12 cores the decoder term scales linearly with items on one
thread and the C5 failure is admission. Cohort formation without waiting is already what
the lockstep loop does inside a worker (B_eff 1.8-2.4 at C4); across workers it would
require moving Talker KV, CP KV, positions, sampling state, decoder state and output
ownership into one process, and the two-submitter results say the decoder cannot then
stay inline. Playback lead as a soft eligibility signal is compatible with global
batching in principle, but the only tested use (parking) failed. The smallest falsifier
before any rewrite is F3: an offline trace of `[ITER]` logs at C4/C5 answering whether
the two workers' steps are coincident often enough that a merged cohort would be B=3-4
rather than an alternation of B=1 and B=2. AMX B ≥ 4 Talker shapes arise naturally only
on hosts with ≥ 3-4 slots per worker; on this host they never will. Global batching
addresses steady-state efficiency on larger hosts, not C5 startup here.

### D. Resumable fixed-prompt prefill after SL-1

| mode | prefill term after SL-1 | PF-1 needed? |
|---|---|---|
| preset / CustomVoice, known text | ~10 positions, ~73 ms flat | no |
| Base/clone with x-vector only, known text | same as preset | no |
| ICL / reference audio | reference frames + reference text overlaid: tens to low hundreds of positions | yes, but the residual set is small and bounded per voice; a per-voice prefix cache of the reference-only positions may remove most of it |
| live incremental text | not a prefill problem; SL-2 parking semantics | no |
| long-form continuation | per-segment re-prefill (~10 positions with SL-1) | no |
| retained non-streaming presets (if F5 fails) | 46-394 ms scaling | yes |

PF-1 is therefore conditional on F5: if the streaming layout passes quality for presets
and clone, the resumable-prefill state machine has expected value only for ICL prefixes
and should be replaced by a per-voice reference-prefix cache; if presets must stay
non-streaming, PF-1 stays on the plan for that mode. Do not carry it forward
unconditionally.

## 6. Native Qwen streaming semantics

Verified (AR-1b, AR-2, and source inspection at the audit HEAD): the official
`non_streaming_mode=False` schedule is known-text; the C implementation matches it
position for position for non-ICL and contains the ICL overlay. Code presence is not
quality qualification. The five things must stay distinct:

1. known complete text with the streaming layout — implemented (SL-1), model-supported,
   quality-gated;
2. live incremental text — model-supported by the same layout with park-not-pad
   semantics; research (SL-2);
3. fixed-prompt resumable prefill — not present; conditional on F5 (section 5D);
4. long-form continuation — decoder-state carry validated externally on this model;
   research;
5. re-prefill/reconstruction — what every span does today.

SL-1 should become a first-class serving path for clone (upstream default) and for
presets if F5 passes. Remaining oracle tests: structural token/position parity against
the official construction for ICL (gate missing), a listening/mel/WER comparison against
the official runtime in streaming mode for presets and clone, and a long-input server
envelope at C3/C4 with SL-1 on.

## 7. External research, targeted

| mechanism | classification | note |
|---|---|---|
| vLLM-Omni per-stream chunk credits (RFC #3535 WS-4) | worth nothing new; unimplemented one-line proposal | the only credit definition is "Stage 0 skips credit=0 streams", i.e. parking — the mechanism we measured and rejected |
| vLLM-Omni adaptive chunk ramp (PR #6001, merged) | transferable conceptually | consumer-side chunk sizing from `buffer_ms`; author found gating on buffer sign "collapses to min_frames under load"; matches our gate result; their ramp floor is unconditional |
| vLLM-Omni EDF re-selection of ready audio streams (PR #6600, in review) | transferable conceptually | ordering among ready work only; no parking; H200 c=64 numbers not predictive for CPU |
| CP stateless re-prefill | contradicted by our measurements/model | bytes unchanged, MACs ×8, barriers unchanged; remains dropped |
| Nari deadline tiers, no batch-forming wait | transferable conceptually | confirms: anchor + fill from ready work; first audio never held |
| X-Square Strategy B chunked prefill | still design-only upstream | nobody has shipped it |
| dynamic initial chunk by load | contradicted for us | our first chunk is 1 frame; our small-call cost is intercept, not left-context recompute; q1/q2/q4/q8 is the primary evidence |
| X2Streaming decoder-state carry | transferable directly (research arm) | structurally present |

Conclusion: external systems corroborate the local finding that lead must order or size
work, never park it, and that first audio is never held for width.

## 8. Interactions between experiments

| interaction | classification | reasoning |
|---|---|---|
| Fused residual shifts the q frontier so q4 or q2 becomes affordable at C4 | UNKNOWN / NEEDS TEST as a causal interaction; SUPPORTED BY MEASURED EVIDENCE for the joint arm | F1 ran fused-on q2/q4/q8 and makes q4 the next C4 playback candidate, but no exact fused-off paired arm was preserved; the earlier control already had q2/q4 below the preferred STREAM p95 target |
| SL-1 reduces the need for prefill slicing enough to refocus scheduling | LIKELY | prefill term 73 ms flat for known text; residual long prefixes are ICL only; quality gate pending |
| Fail-fast creates a clean place for startup priority / utilization-aware admission | SUPPORTED as structure | the parent already sees every arrival when `--max-queue 0`; an admission test can sit exactly there |
| Global batching would preserve cohorts better than the lead gate | PLAUSIBLE | cohorts are what the gate destroyed; but global batching adds inline decoder items (contradicting term) |
| Consumer failed partly because ownership changed at the same time batching disappeared | SUPPORTED | `dec_batch = 0` is forced by the flag (2818); group=1 observed |
| OUT isolation makes a future admission/scheduler safer though KPI-neutral | SUPPORTED | writes off the engine thread; slow-reader isolation proven |
| MT-4 revealed that earlier startup conclusions were transport artifacts | CONTRADICTED for C≤4, SUPPORTED for C5 | at C3 TTFB 61 ms vs TTFA 167 ms: startup was real synthesis; at C5 the tail is pre-admission and was invisible before MT-4 |
| Raising raw decoder throughput alone cannot make C5 interactive | SUPPORTED | C5 STREAM 0.852 with TTFB 4.28 s; the wait is a slot |
| Reducing decoder fixed cost permits smaller q without losing STREAM | LIKELY | the intercept term is what makes q2 cost 31.5 vs 22 ms; every intercept cut moves the floor down |
| q2/q4 are more attractive after fused than in the original sweep | UNKNOWN / NEEDS TEST as a delta; SUPPORTED for fused+q4 now | q4 has the best current playback/realtime Pareto point, but F1 cannot attribute the change to fusion versus run/bank differences |
| Lead as a sizing signal (quantum) succeeds where lead as a parking signal failed | PLAUSIBLE | vLLM-Omni's merged controller is exactly this; untested locally |
| A third slot per worker is affordable with fused + q4 | UNKNOWN | B3 Talker and CP never measured |

## 9. Architecture decision matrix

| mechanism | current evidence | architectural value | risk | decision | next action |
|---|---|---|---|---|---|
| SL-1 known-text layout | prefill flat 73 ms; server path proven; quality/ICL gates open | high (first play, admission hole) | model-visible quality | PROMOTE (deployment-selected per mode after F5) | F5 |
| async output isolation | isolation proven; KPI-neutral; memory ~9 GB PSS both arms | high (correctness) | thread/memory at long C | KEEP, then default after 5-min C4 qualification | qualification run |
| MT-4 | proven | metric truth | none | PROMOTE (done) | — |
| q2/q4/q8 quantum | F1 fused-on C4 screen: q4 is the best current playback/realtime candidate; q8 is the throughput control | high (cadence lever) | throughput at small q; no long SOAK yet | PROMOTE q4 as next C4 reference candidate, not production qualification | F2; later qualification |
| hard lead gate | rejected; 95.8 % suppression | none as parking | — | REJECT (keep code as diagnostic) | — |
| prefill helper | rejected; TTFA p95 2.4 s | none | — | REJECT as serving | — |
| true resumable prefill (PF-1) | not built; need narrowed by SL-1 | low-medium (ICL only if F5 passes) | complexity | DEFER, conditional on F5; prefer per-voice reference-prefix cache | after F5 |
| same-pool decoder consumer | rejected; loses gang, serializes | none in this form | — | REJECT (keep flag diagnostic) | — |
| ragged scratch reuse | reverted | none | lifetime | REJECT | — |
| fused residual | C4 A/B and SOAK positive | high | quality on other voices | PROMOTE (deployment-selected → default after quality bank) | quality bank + per-class SOAK |
| admission fail-fast | proven; selection effect only | high (semantics) | none | PROMOTE (deployment-selected) | write into profile |
| utilization-aware admission (LS-4 concrete) | not built; cause SUPPORTED | high (C5 path) | worker→parent feedback | REWORK LS-4 into this | F2, F-cap3 |
| global Talker batching | B2/B1 real; inline decoder term contradicts on 12 cores | medium on larger hosts | rewrite, ownership | DEFER (host-conditional) | F3 offline trace |
| global CP batching | same-codebook batching exists per worker | low here | — | DEFER with the above | — |
| AMX B ≥ 4 Talker/CP | shapes never arise on this host | host-conditional | — | DEFER | — |
| structural decoder rendezvous/intercept reduction | fused shows the class works; counters missing | high (moves the q floor) | numerics per change | KEEP as the P4 continuation, counters first | pool-submit/wait counters |
| decoder strip executor | not built | medium (intercept) | large | DEFER behind counters and F1 | — |
| long-form decoder state carry | externally validated on this model | medium (product) | quality | RESEARCH | CLI falsifier |
| live incremental text | model-supported; parking rule known | medium (product) | in-distribution unknown | RESEARCH | CLI falsifier |
| static core lanes | no evidence; Talker needs full bandwidth at B ≤ 2 | none now | — | REJECT for this host | — |

## 10. Next falsifiers (six, in order)

**F1 — fused residual × decoder quantum — COMPLETE.** This was a flags-only C4
three-wave screen on the exact AMX reference: fused on, warm strip, q2/q4/q8, short
diverse bank, seed base 2027, synchronous output, no profiler/census. The clean AMX
build was from HEAD `e60aebf65cd8d068c7ff89c2a5a0cd8e5511cd6a`; binary prefix
`9d8052daf7e7b01f`; receive coalescing was 0 % for every arm. q4 reached STREAM p95
0.868, required prebuffer p95 201 ms and stall@250 0 %, with TTFA p95 175 ms. q2
reached STREAM p95 0.914 and q8 reached 0.822; q8 had the higher throughput but
required prebuffer p95 306 ms and stall@250 25 %. No errors, rejects, timeouts or
service-cap events occurred. q4 is therefore the next C4 playback/realtime reference
candidate; it is not a five-minute qualification or a production-default decision.
The screen establishes the fused-on joint frontier, not a causal fused-vs-off delta:
the exact paired fused-off anchor was not run, and the earlier non-identical control
already had q2/q4 below the preferred p95 target. Next: F2, not another q value.

**F2 — C5 startup decomposition with parent timestamps.** Hypothesis: the C5 tail is
entirely pre-`t_recv` slot wait and a fifth admitted request is otherwise normal.
Minimal implementation: parent-side `accept` and dispatch timestamps propagated to the
`[PATH]` trace (a few lines; diagnostic only). Control: reference config, fused on,
default queue policy, C5 three waves; second arm `--max-queue 0`. Metrics: per-request
backlog wait, parent admission, child pre-service, prefill, first step, first decode,
first PCM; plus the envelope. Promotion: backlog wait explains > 90 % of TTFB p95 and
post-admission terms match C4. Falsifier: material time inside the child before first
PCM → admission is not the only C5 problem. Confounders: same bank and waves as the
C5 screen.

**F-cap3 — a third slot on one worker.** Hypothesis: a B=3 worker with fused + the F1
quantum is at least MARGINAL and the fifth request becomes interactive. Minimal
implementation: none (`--batch-size 3` on the server; parent cap follows). Control: F1
winner at C4 (cap 2). Arms: cap 3 at C5 and at C4 (to see whether cap 3 harms C4 by
letting one worker take three). Metrics: full envelope plus per-worker B and
`[ITER]` step wall at B=3. Promotion: C5 envelope GOOD or MARGINAL with established
streams GOOD. Falsifier: STREAM p95 > 0.95 or stall@500 > 5 % at C5 → a third slot is
unaffordable on 12 cores; C5 needs cost reduction first. Confounders: fused on in both
arms; no other knob.

**F3 — offline cohort coincidence trace.** Hypothesis: real C4/C5 traffic rarely has
both workers stepping B=2 simultaneously, so a merged engine would mostly alternate
B=1/B=2 rather than reach B=3-4. Minimal implementation: parse existing `[ITER]` logs
(or one C4/C5 diagnostic wave with `QWEN_TTFA_TRACE`) for per-worker `n_active` per
iteration and cross-worker overlap in time. Control: none (read-only). Metrics:
distribution of merged-B per 80 ms window. Promotion: merged B ≥ 3 in > 50 % of windows
at C5 → global batching has a cohort to amortize. Falsifier: < 25 % → DEFER stays.

**F5 — SL-1 quality closure.** Hypothesis: the streaming layout is quality-equivalent
for presets and clone. Minimal implementation: ICL smoke assertion; a run of the
official runtime in streaming mode on a box with the reference assets. Control:
non-streaming C output and official streaming output, same seeds/texts. Metrics:
mel-corr, WER, speaker similarity, ear check; prefill and TTFA versus length on the
server at C3/C4. Promotion: parity within the golden gate → SL-1 deployment-selected
for that mode, PF-1 dropped for it. Falsifier: worse on presets → presets stay
non-streaming and PF-1 stays for them.

**F6 — utilization-aware admission prototype.** Only after F2 and F-cap3. Hypothesis:
admitting a third slot when the target worker's EWMA iteration wall at B+1 fits under
`U_max`, else 503, yields interactive C5 for affordable arrivals and honest rejection
otherwise. Minimal implementation: worker reports its last iteration wall per B on the
existing done-byte channel or a shared word; parent applies the test. Control:
`--max-queue 0` with cap 2. Metrics: accepted-set envelope, rejects, established-stream
stall during admissions. Promotion: accepted fifth requests GOOD, established streams
unchanged. Falsifier: established streams degrade whenever a third slot is admitted.

Deferred without a clean falsifier: soft lead eligibility (F4). Its only mechanically
different form is lead-sized quantum selection, which F1 prepares and LS-2 implements;
lead-ordered decode among ready items has no choice to make at B ≤ 2.

## 11. Evaluation of the proposed high-value falsifiers

F1 completed: q4 is the strongest current C4 playback/realtime candidate, q8 remains
the throughput control, and the fused interaction is not causally isolated. F2 accepted
with the parent-timestamp addition. F3 accepted as read-only. F4 deferred (no bounded
mechanism other than LS-2). F5 accepted, conditioned on asset availability. Added:
F-cap3 and F6, because the C5 evidence points at slots rather than scheduling.

## 12. Cross-ISA implications

ISA-independent semantics (freeze before VNNI/ARM parity): parent fail-fast and
utilization-aware admission; whole-connection slot accounting; output ownership and
backpressure semantics; the known-text streaming layout and trailing-text state;
cancellation/disconnect behavior; the lockstep loop with inline ragged decode as the
cohort mechanism; lead used for sizing/ordering, never parking; metric definitions.

ISA-dependent policy (re-measure per host): decoder quantum (intercept/slope differ),
minimum efficient quantum, per-worker slot cap, Talker/CP batch thresholds, pool
width, ragged threshold, target lead, U_max for admission.

ISA-specific implementation: Design-D INT8 AMX panels and the fused residual epilogue;
VNNI decoder kernels; KleidiAI/i8mm kernels. The fused-residual gain must not be
assumed on VNNI or Axion; each needs its own A/B under the same harness.

## 13. Final proposed architecture (not current runtime; unvalidated)

One incremental architecture is proposed here; every element is either already in the
tree or bounded by a falsifier above. This section is a design hypothesis, not a claim
about the defaults or behavior of the audit HEAD.

**Request admission.** The current explicit fail-fast arm (`--max-queue 0`) accepts a
full-listener connection so the parent can return 503; the default still gates the
listener while all slots are occupied. A future utilization-aware policy could route
to the least-loaded worker only when a measured B+1 iteration fits its budget and
otherwise fail fast. No `U_max` policy exists yet. Slot occupancy remains
whole-connection until a bounded capacity test shows otherwise.

**Ownership.** Prefork workers keep full ownership of their slots (Talker KV, CP KV,
decoder stream state, trailing text, output queue). No state moves across processes on
this host. The only new cross-process signal is the worker's iteration-wall report to
the parent.

**Startup path.** Known text uses the streaming layout (per mode after F5): prefill
~10 positions, first chunk one frame, header at admission. ICL clone keeps its
reference prefix, optionally served from a per-voice prefix cache. Inline prefill is
retained; its hole is 73 ms for known text and bounded by the reference length for ICL.

**Talker/CP scheduling.** The lockstep loop is the cohort: all active slots step
together every iteration; admissions join the next step; nothing waits for width;
nothing parks. Lead is not consulted here.

**Decoder scheduling.** Inline ragged gang retained; fused residual on; quantum chosen
per stream from lead (LS-2): q4 is the current F1 C4 candidate for streams below target
lead, while q8 remains the throughput control; q2 is a mandatory-realtime alternative
but missed the preferred STREAM p95 target in F1. First chunk always one frame; ramp
floor never shrinks under load. No second submitter, no overlap, no intra-call
preemption. Intercept/rendezvous reduction continues as the lever that lowers the
affordable floor, with pool-submit/wait counters added first.

**Output.** Detached per-stream writer with byte cap, send timeout, fail-and-close on
overflow; conversion stays on the engine thread until measured otherwise.

**Metrics.** Policy inputs: per-worker iteration wall per B (admission), per-stream lead
(quantum). Diagnostics only: client-observed prebuffer/stall, pool counters, phase
timers, core-equivalents.

**Overload.** At C5/C6 on this host, the next falsifier is whether a third slot is
affordable (F-cap3). Until that is measured, the design must not promise admission;
the already-proven fail-fast arm can reject within milliseconds, while the default can
still expose pre-admission backlog delay.

**Future scaling.** On a host with ≥ 3-4 affordable slots per worker the same semantics
hold and Talker/CP cohorts reach B ≥ 3-4 naturally, which is where AMX Talker/CP and a
single-engine ready set become worth testing (F3 first). On VNNI/ARM only the policy
constants and the decoder kernels change.

Unknown after this design: B=3 costs, the causal fused-residual delta on the quantum
frontier, whether q4 survives a longer qualification, SL-1 quality, natural cohort
coincidence, and the true pool-wait share of the decoder call.

# Codex handoff

**1. Recommended architecture** (section 13): keep the lockstep inline-ragged worker,
add utilization-aware fail-fast admission at the parent, streaming layout for known
text per mode, fused residual, lead-sized decoder quantum, detached output. No second
submitter, no parking, no global engine on this host. Status: STRONG INFERENCE for the
whole; components individually labeled below.

**2. Evidence supporting it**
- Critical-path shortening wins, concurrency-adding loses (section 4, H1/H2): SUPPORTED.
- C5 failure is pre-admission slot wait (TTFB p95 4.28 s, identical at C5/C6, child
  pre-service < 200 ms): SUPPORTED.
- Fused residual C4 A/B and SOAK: SUPPORTED.
- F1 fused-on quantum screen: q4 is the best current C4 playback/realtime candidate;
  q8 remains the throughput control. The fused-vs-off frontier shift is UNKNOWN because
  no exact paired off arm was run.
- SL-1 flattens prefill for known text: SUPPORTED (mechanism); quality: UNKNOWN.
- Lead as parking is harmful in this loop: SUPPORTED; lead as sizing: WEAK INFERENCE
  (external precedent only).

**3. Rejected alternatives and why**
- Same-pool decoder consumer: serializes on `submit_mtx`, drops the ragged gang (forced
  `dec_batch = 0`), TTFA and max-gap ×6-7. REJECT this realization.
- One-shot prefill helper with LOW priority: starves on a busy pool; TTFA p95 2.4 s.
- Hard lead gate: parks 95.8 % of steps; loop sleeps; STREAM → 0.99 by construction.
- Ragged scratch reuse: neutral/negative; reverted.
- Global Talker/CP engine on 12 cores: inline decoder term scales with items; C5 is not
  a compute problem; DEFER host-conditionally.
- Static core lanes; CP stateless re-prefill; q1; q32; naive sentence splitting.

**4. Assumptions Codex must verify against current code**
- Parent cap binds before the child grace slot, so `--queue-timeout-ms` never sees the
  backlog wait (`qwen_tts_server.c:2244`, `:2583`, `:2719`, `:2789`).
- `QWEN_DECODER_THREAD=1` forces `dec_batch = 0` (`qwen_tts.c:2818`).
- Lead gate sleeps the whole loop when no slot is eligible (`qwen_tts.c:3262-3264`).
- Quantum policy has no floor in code (`qwen_tts.c:3338-3344`).
- Fused residual changes the ragged conv2 route FP32 → INT8 (`qwen_tts_speech_decoder.c`
  ragged gate ~3357-3371, fallback 3699-3705); per-slot route unchanged.
- SL-1 ICL overlay exists (`qwen_tts.c:1342-1390`) but no ICL gate runs.
- The reference serving configuration has no committed profile; defaults differ
  (`QWEN_SD_RAG_MIN_PANELS` 8, `QWEN_SD_AMX_D` off).
- Async output header is written by the writer at admission; conversion and `malloc`
  remain on the engine thread.

**5. Semantic/model risks**
- SL-1 is model-visible; quality for presets is UNKNOWN; upstream defaults differ by mode.
- Fused residual ragged route is a numerical change; only the tested CLI contract is
  byte-identical; other voices/models UNKNOWN.
- Live text park-not-pad and long-form carry are research; in-distribution behavior
  UNKNOWN.

**6. Top next falsifiers in exact order**
F2 (C5 decomposition with parent timestamps) → F-cap3 (third
slot) → F5 (SL-1 quality closure) → F3 (offline cohort trace) → F6 (utilization-aware
admission prototype).

**7. What must NOT be implemented yet**
Utilization-aware admission before F2/F-cap3; LS-2 lead-sized quantum before a longer
q4 qualification;
PF-1 before F5; any global engine or cross-worker state move before F3; any decoder
consumer or overlap variant; any lead-parking variant; strip executor before
pool-submit/wait counters exist; VNNI/ARM ports of the fused epilogue.

**8. PLAN items to promote/rework/defer**
- PROMOTE: fused residual to deployment-selected (profile), MT-4 (done), fail-fast as
  a profile option, a committed reference profile for 2x6/cap 2/q8/threshold 2/Design-D.
- REWORK: LS-2 into "lead-sized quantum from the F1 frontier"; LS-4 into
  "utilization-aware fail-fast admission at the parent" with F2/F-cap3/F6; the P4 open
  item into "counters first, then intercept/rendezvous reduction".
- DEFER: PF-1 (conditional on F5), EO-1/EO-2 (conditional on F3 and host), AMX B ≥ 4.
- KEEP: OUT-1 default-off until the 5-minute C4 thread/memory qualification; SL-1
  default-off until F5, then per-mode.
- Correct PLAN text: "blocking writes, no send timeout" (sync path only); "~41
  rendezvous" (call-site inventory); SL-1 ICL "implemented, ungated".

**9. Exact files/functions to inspect first**
`qwen_tts_server.c`: `qwen_tts_serve_prefork` (2566-2800), `jq_push` (1572),
`sink_next_job` (1806-1841), `sink_step_allowed` (1924), `stream_output_enqueue`,
`set_client_timeout` (1373). `qwen_tts.c`: `qwen_tts_serve_continuous` (2629; gate
3249-3267; quantum 3338-3344; gang 3457-3507; `dec_batch` 2818), layout branches
(1241-1405), `RECORD_FRAME_AND_EMBED` (3069-3086), `dec_worker_main` (2525).
`qwen_tts_speech_decoder.c`: fused gates (~2096-2109, ~3357-3371), `sd_rag_panel_worker`
(3090), batch dispatch (3798). `tests/serve_parallel_wave.py` wave semantics (570-580).
`configs/perf/gcp-c4-standard-24-vnni-ttfa.json`.

**10. STOP conditions**
- F1 produced a q4 candidate, but it is not yet a qualification: run the next bounded
  F2 falsifier before implementing LS-2 or calling q4 production-ready.
- F2 shows material child-side time before first PCM at C5 → stop the admission-only
  hypothesis; re-decompose.
- F-cap3 shows established streams degrade at cap 3 → stop C5 on this host; document C4
  as the ceiling and move capacity work to larger hosts/other ISAs.
- F5 fails for presets → SL-1 clone-only; PF-1 stays for presets.
- Any experiment with errors, rejects (outside fail-fast arms), timeouts or receive
  coalescing above ~10 % is not evidence.

## Conclusion

The accumulated evidence is consistent with one lockstep loop per worker whose inline
decoder burst is a major measured term and whose C5/C6 startup tail is dominated by
pre-admission slot occupancy in the tested arms. Shortening material critical-path work
helped; adding threads, parking work or moving one-shot prefill hurt. F1 now gives q4 the
best measured C4 playback/realtime envelope under the fused-on configuration, while q8
remains the higher-throughput control. It does not prove a causal fused-vs-off frontier
shift, decoder arithmetic or memory pressure negligible, idle cores freely exploitable,
or a third slot affordable. The smallest coherent architecture therefore remains a
falsifiable proposal: qualify q4 with F2/longer C4 evidence, qualify SL-1 by mode, and
measure admission capacity before making it utilization-aware.

## Next action

Run F2, the C5 startup decomposition with parent timestamps, using q4 as the next C4
reference candidate. Do not implement section 13 or run F-cap3 before F2 reports.
