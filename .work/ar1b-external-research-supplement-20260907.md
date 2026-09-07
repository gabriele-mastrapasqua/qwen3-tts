# AR-1b · External research and architecture supplement

Task: AR-1 (supplement; read-only research phase, no code, no PLAN edit, no local benchmark)

Frozen checkpoint: `8d9ff3de9bb75009d42d43c3875925c16e9e2540`, branch `feature/x86-amx-vnni-oss`.
Inputs: `.work/p2-checkpoint-20260907.md`, `.work/ar1-post-p2-architecture-review-20260907.md`
(AR-1), `.work/ar1-codex-implementation-audit-20260907.md` (authoritative for runtime facts),
`.work/professional-streaming-architecture.md`, `.work/post-p2-streaming-research-agenda.md`,
`PLAN.md`. Six parallel research passes were run (official Qwen3-TTS semantics, vLLM-Omni
source, alternative runtimes source, long-form literature, scheduling theory, local CP
audit); every source claim below carries its URL and the hardware/workload of any number.

Codex corrections preserved throughout: 102 is a source-level call-site inventory, not a
rendezvous count; pool submissions and waits are UNKNOWN without counters; the prefix-cache
nonzero position is not a resumable prefill; fixed-prompt incremental prefill needs a new
request-owned cursor/state machine; q2/q4 are smaller complete decoder calls, not intra-call
preemption; true preemption needs persistent stage/strip/cursor/intermediate state;
same-pool overlap is plausible but unproven; 7.3-7.4/12 cores does not prove exploitable
idle capacity; B2/B1 proves an amortization opportunity, not a global-engine design;
appending text after acoustic generation is a model/layout question.

## Question

Which mechanisms already demonstrated in TTS, multimodal and long-form codec-LM serving
inform the next CPU architecture, and which of them actually transfer to this Qwen3-TTS C
engine? For each: external mechanism → problem it solves there → local bottleneck → what
transfers → what does not → minimal local falsifier → PLAN impact.

## Known facts (new, from sources; local facts are in AR-1 and the Codex audit)

- The official Qwen3-TTS streaming layout (`non_streaming_mode=False`) is a per-position
  dual-track schedule: prefill ends at `text[0] ⊕ codec_bos`; generated frame k is fed
  back summed with `trailing_text[k]` (text[1..N−1], then `tts_eos`, then `tts_pad`).
  One text token is consumed per codec frame with one-token lookahead. The official Base
  clone path uses this layout by default; CustomVoice/VoiceDesign default to
  non-streaming. The docstring says the flag "only simulates streaming text input"
  because the whole text is known and the schedule is fixed at prefill
  (`qwen_tts/inference/qwen3_tts_model.py` L513-515, L478, L642, L738;
  `modeling_qwen3_tts.py` L1671-1686, L1968-2260;
  https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/).
- The technical report (arXiv:2601.15621) states the dual-track design is for "streaming
  text input and audio output" and "upon receiving a textual token, the model immediately
  predicts the corresponding acoustic tokens", trains up to 32,768 positions, and gives
  no alignment/lookahead/mask description. Maintainers (issue #10) say streaming is
  "supported at the model architecture level" and defer serving to vLLM-Omni; issues #77
  and #207 asking for the training alignment were closed without answer.
- Two independent source-level Qwen3-TTS runtimes implement the official streaming
  layout for live text and converge on the same rule: when live text is exhausted,
  park the Talker (keep KV, feed nothing), never feed `tts_pad` (it means "text finished"
  and triggers ending), never restart. Nari: `TextContinuation.has_token` gates
  `TALKER_DECODE`, EOS suppressed while input unfinished
  (https://github.com/nari-labs/nari-qwen3-tts, `contract/request.py` L293,
  `planner/planner.py` L161-168). X-Square: `slot.next_embed = None`, `last_codec_sum`
  retained, evicted after 10 s idle (`engine/backend/engine_loop.py` L1944-1962,
  https://github.com/X-Square-Robot/Qwen3TTS-Streaming).
- Nari (H100 only, FP8, CUDA graphs) uses a 1-frame first codec chunk in its `ttfa`
  profile with schedule `[1,2,4,8,12]` frames, `[4,4,8,16,25]` balanced, `[25]`
  throughput; a stateful incremental codec (conv histories, transposed-conv overlaps,
  pre-transformer sliding KV, no lookahead, 1 frame → 1920 samples); a `deadline_aware`
  policy with per-request deadline `playback_started_at + emitted_duration` and tiers
  (codec work within a 10/25 ms reserve, then startup work, then "pressing" work within
  1.0 s of its deadline); per-request 8 MiB PCM buffer → cancel and close 1011 on
  overflow; sender 32 frame slots, 5 s send timeout. Claims 10 RPS p95 TTFA < 50 ms on
  H100 (`profile.py`, `planner/policy.py`, `model/incremental_codec.py`,
  `contract/stream.py`).
- X-Square (RTX 5090, TensorRT fused step = Talker + CP + Code2Wav for one frame):
  continuous batching on one engine thread, priorities FIRST_SEGMENT < CONTINUATION <
  PREFETCHED plus an MLFQ for decode ordering; long text split at punctuation tiers
  using an online EMA of frames per text token, each segment an independent prefill,
  only the text tail carries (`kv_cache_pool.reset_for_new_segment()` has no call
  sites; `last_segment_codec_tail` "Reserved"); Strategy B (prefill chunks between
  decode steps) is six lines of design and an unchecked roadmap item; Strategy A
  (serial prefill at step boundaries) is what runs (`docs/dev/architecture.md`
  L994-1000, L2236; `engine_loop.py` L410-415). Single-stream TTFT 14.9 ms and
  12.6 ms/step, c=128 TTFT 242-275 ms at 42.1 ms/step, RTX 5090, 100 % prefix-cache hits.
- X2Streaming-TTS (arXiv:2608.18661, X-Square, built on Qwen3-TTS 1.7B, inference-only):
  carries the complete Code2Wav state bundle (KV, conv states, transposed-conv states,
  frame index) across text segments plus H=4 trailing Talker states injected as a
  bounded memory (gain ≤ 0.015); punctuation-tier segmentation with capacity-adaptive
  limits; PBD 0.1092 vs 0.19-0.35 for chunk-based baselines, ECAPA SIM 0.951, ΔE
  1.66 dB at boundaries; TTFT 15.8 ms (1 request) to 261 ms (128 concurrent) on one
  RTX 5090.
- Prosodic-boundary-aware streaming (arXiv:2603.06444, CosyVoice2 base): untrained
  slide-and-prompt continuation (prompt = previous chunk's text + codes, reference
  dropped) collapsed speaker similarity to 0.22; after boundary-aware post-training
  0.65 and long-form WER 4.77 % vs 70.97 % for unbounded interleave (A40).
- MagpieTTS-LF (arXiv:2606.18485, NeMo): carries the last K=20 text tokens, their
  encoder hidden states and the last attended text position into a soft cross-attention
  prior; decoder restarts per sentence; no acoustic state carried; inference-only;
  WER 0.025 vs Qwen3-TTS 0.045 on 20 texts of 3-4 min, A6000; encoder-decoder specific.
- VALL-E (arXiv:2301.02111): causal LM training makes any prefix a prompt;
  VALL-E-continual (first 3 s + full transcript) WER 3.8 vs 5.9 but speaker similarity
  0.508 vs 0.580; one utterance, no windows; V100 cluster. Background evidence only.
- vLLM-Omni (upstream H200/H20 reference): Talker and Code2Wav are separate engines
  joined by a shared-memory connector with an unbounded chunk deque and no credit, ack or
  window from Code2Wav back to the Talker; the producer is never throttled (issue #4913:
  3-4 s chunk gaps at 8 concurrent on Ascend; #4855: GPU busy 12-13 % of wall at c=64, the
  step is host-side Python). Dynamic initial chunk `[2,4,8,16]` picks by
  `active_requests/max_num_seqs` of the Talker (not Code2Wav load); the shipped yaml pins
  `initial_codec_chunk_frames: 1`. An opt-in `AdaptiveChunkController` computes
  `buffer_ms = emitted − elapsed` (our playable lead) and sizes the next chunk as the
  largest n with `n×ewma + margin ≤ buffer_ms`, but refuses to shrink under load because
  its decoder re-decodes 72 frames of left context per emit (n=2 costs ~10× n=25). The
  Code Predictor re-prefills the whole ≤16-token sequence with SDPA and no KV cache
  because "removing the KV cache machinery saves far more time than it costs" on GPU
  (PR #1617: ~27 % RTF on RTX 5090); it keeps RMSNorm/RoPE/attention in fp32 after
  numerics compounding across 15 steps degraded UTMOS 4.26→2.66 (#2274). Streaming text
  input is `input.append`/`input.done` = buffer then synthesize as one request; the
  early-start `start_policy` is an unimplemented RFC (#1951, #2115). H200 c=1 TTFP
  733→64 ms with "~30 % more compute", RTF 0.124→0.160.
- Local CP audit (agent F, this HEAD): per frame 16 transformer positions × 5 layers,
  78.6 M MAC per position, 1.32 GMAC/frame and 1.32 GB of INT8 weight reads per frame
  (L3-resident, ~86 GB/s per worker at C1); KV traffic 0.23 % of bytes; 718 barriers per
  frame in the batched region (8 per layer × 5 × 16 plus heads); stateless recompute
  keeps weight bytes constant and raises MACs 8.1× (≈1.7-1.8× projection time by the
  local row-cost model), removes no barrier; CP KV allocates 64 slots of which 16 are
  used.

## Unknowns

- Whether feeding `tts_pad` while text is still pending, then resuming text, is
  in-distribution (no runtime does it; both park instead).
- Quality delta between streaming and non-streaming layouts for CustomVoice presets
  (official default is non-streaming for presets, streaming for clone).
- Whether the C engine's ICL clone output matches the official default (streaming) clone
  layout; today the C path implements only the non-streaming layout for both.
- Whether decoder-state carry across Talker segments without the H=4 Talker memory
  (i.e. the shipped X-Square behavior plus our persistent decoder state) is
  audibly acceptable on this model; X2Streaming measured the combination, not the parts.

## Files/functions inspected

Local: `qwen_tts.c` prompt assembly (1094-1337) and step embedding (1679-1689),
`qwen_tts.h` decoder stream state (293-320), `qwen_tts_code_predictor.c` (804-892,
1042-1115, 1282-1356, 1377-1577), `qwen_tts_server.c` output path (1497-1530), the
Codex audit. External: files cited inline.

---

## 1. Executive research verdict

Three things change relative to AR-1 plus the Codex audit; everything else confirms them.

1. **The long-prefill problem is largely a layout problem, not a scheduler problem.**
   In the official streaming layout the Talker prefill ends at `text[0] ⊕ codec_bos`
   (about 10-12 positions for a preset voice) and the remaining text is consumed one
   token per generated frame. Prefill wall and the admission hole therefore stop
   scaling with text length; the 46 → 394 ms growth measured in
   `.work/p2-input-length-scaling-20260907.md` is a property of the non-streaming
   layout the C engine implements exclusively. This is model semantics (trained
   dual-track, official default for clone, implemented by Nari and X-Square), not a
   serving trick. It is the highest-leverage mechanism AR-1 and the audit both missed,
   and it is cheaper than a prefill state machine. It does not remove the need for a
   bounded prefill budget for the residual long prefixes (ICL reference frames,
   non-streaming mode kept for presets if quality requires).
2. **The serious runtimes bound decoder bursts with small complete stateful calls, not
   intra-call preemption.** Nari issues one decode step per planner decision and a
   1-frame first codec chunk on a stateful incremental codec; X-Square fuses one frame
   per step. Neither yields inside a decoder call. Combined with the Codex proof that
   our decoder is not intra-call resumable, P3 should start with smaller complete calls
   under lead scheduling (Option 1 of Q8), and the "completed-stage state machine" is
   throwaway unless the overlap experiment demands it. The strip executor's role is to
   make small calls cheap (intercept), which on GPU is hidden by CUDA graphs and on CPU
   is the 23-28 ms intercept measured in CT-2.
3. **Credit-gated EDF on audio lead is the scheduling primitive, and vLLM-Omni's
   unthrottled producer is the negative lesson.** Established theory (Liu & Layland,
   EEVDF/DRR eligibility, CBS reservations, RED/Clockwork admission) plus two field
   precedents (Nari's deadline tiers, vLLM-Omni's buffer-feedback chunk controller)
   converge on: eligibility by target lead, ordering by lead deadline, a reserved
   prefill budget per iteration, admission by predicted laxity, shed-newest under drift.
   On shared LLC/DRAM a stream 3 s ahead spends bandwidth a stream at 0 s needs, so
   suppressing Talker steps above target lead is a CPU-specific gain the GPU systems
   never needed.

Two ideas are closed by evidence: CP stateless re-prefill (bandwidth-bound CP, bytes
unchanged, MACs ×8, barriers unchanged) and untrained slide-and-prompt continuation with
the model's own codes as the only speaker anchor (measured collapse to SIM 0.22 on a
comparable decoder-only model).

## 2. Source map

| source | mechanism demonstrated | relevance to our engine | confidence |
|---|---|---|---|
| Qwen3-TTS report 2601.15621 + official code | dual-track per-position streaming layout; one text token per frame; wait/pad/eos semantics; ICL prefix as codes ⊕ text overlay | DIRECTLY TRANSFERABLE: the C engine implements only the non-streaming layout; four localized differences (agent A) | high (code read) |
| Nari (`nari-qwen3-tts`, H100) | wait-not-pad live text; 1-frame first chunk + ramp; stateful incremental codec; deadline tiers; drop-and-close output | TRANSFERABLE MECHANISM, DIFFERENT IMPLEMENTATION for scheduler/output; DIRECTLY for the text rule and the ramp | high (code read) |
| X-Square `Qwen3TTS-Streaming` (RTX 5090) | parked slot with `last_codec_sum`; priority tiers + MLFQ; punctuation-tier windowing with EMA frames/token; Strategy B chunked prefill is design-only | DIRECTLY TRANSFERABLE for windowing rule and parking; INSUFFICIENT EVIDENCE for Strategy B | high (code read) |
| X2Streaming-TTS 2608.18661 (Qwen3-TTS 1.7B, RTX 5090) | Code2Wav state carry across segments + H=4 Talker memory, inference-only, boundary metrics | DIRECTLY TRANSFERABLE for decoder-state carry (our `qwen_sd_stream_state_t` is that bundle); TRANSFERABLE MECHANISM, DIFFERENT IMPLEMENTATION for Talker memory (forward change) | medium (paper; KV reset unspecified) |
| Boundary-aware streaming 2603.06444 (CosyVoice2, A40) | untrained slide-and-prompt collapses SIM; post-training fixes it | NOT TRANSFERABLE as a recipe; decisive negative evidence for own-codes-only re-prompting | high |
| MagpieTTS-LF 2606.18485 (A6000) | text-history + encoder-state + attention-prior carry; decoder restart per sentence | NOT TRANSFERABLE (encoder-decoder); the "carry K text tokens" half is INTERESTING ANALOGY ONLY | high |
| VALL-E 2301.02111 | any prefix is a prompt under causal training; continual WER better, SIM worse | background; the prefix-as-prompt property is TRANSFERABLE MECHANISM, DIFFERENT IMPLEMENTATION | high |
| vLLM-Omni 2602.02204 + source (H200/H20) | stage split with unthrottled producer; dynamic IC by Talker load; lead-based adaptive chunk controller; CP re-prefill; fp32 CP numerics; continuous batching without batch-forming delay; abort purge | mixed: adaptive chunk signal DIRECTLY; stage split NOT; CP re-prefill NOT (see §9); unthrottled producer is a counter-example | high (code read) |
| Scheduling theory (Liu & Layland; EEVDF; DRR; CBS/SCHED_DEADLINE; RED; Sarathi-Serve; Clockwork; Kairos) | EDF optimality and its overload domino; eligibility gates; reservations; admission by predicted completion | DIRECTLY TRANSFERABLE as the primitive in §8 | high (established theory) |
| NetEq / JACK / GStreamer | target-delay set point with hysteresis, not maximum buffering; latency admission | DIRECTLY TRANSFERABLE for lead target semantics | high |
| dffdeeq / rekuenkdr forks (RTX 5090) | re-decode-and-crop 48-80 frame windows | INTERESTING ANALOGY ONLY; our stateful decoder already avoids it | high |
| CosyVoice2 / IST-LM / SpeakStream / DSM / VoiceChat-TTS / FireRedTTS-2 | trained interleaves for live text; KV-carry speaker drift over turns | INTERESTING ANALOGY ONLY (training-defined schedules); the drift result is a warning for unbounded live sessions | medium |

## 3. Native Qwen3-TTS streaming semantics

Model semantics (trained, evidenced by the official code and report): the Talker input at
each position is the elementwise sum of a text-track embedding and a codec-track
embedding sharing one position id. Non-streaming: `[role][control ⊕ tts_pad…tts_bos]
[text ⊕ codec_pad … tts_eos ⊕ codec_pad][tts_pad ⊕ codec_bos]`, then every generated
frame is fed back as `codec_sum ⊕ tts_pad`. Streaming: the prefill ends at
`text[0] ⊕ codec_bos`; generated frame k is fed back as `codec_sum_k ⊕ trailing[k]`
with `trailing = text[1..N−1], tts_eos`, then `tts_pad`. ICL streaming overlays
`[ref_text, text, tts_eos]` on `[codec_bos, ref_frame_0, …]` and keeps the overflow as
the trailing list. EOS is codebook-0 `codec_eos`; `min_new_tokens=2`.

Offline simulation: the official inference computes the whole trailing schedule at
prefill because the full text is known; nothing in the released code accepts text later.
That is the meaning of "simulates". The mechanism itself is a fixed per-step rule and
needs no simulation to be applied online: an engine can look up `trailing[k]` from a
buffer that grows as text arrives.

Unknown: what the model does if the text buffer is empty before `tts_eos` was emitted.
Feeding `tts_pad` means "finished" (both runtimes avoid it). Parking the Talker (skip the
step, retain `codec_sum`) is the observed practice and is semantically neutral (the KV
does not advance). Text normally arrives faster than 12.5 tokens/s from any producer, so
parking is rare after the first token.

Local comparison (agent A, `qwen_tts.c`): the C engine implements only the non-streaming
layout for preset, design and ICL clone. The streaming layout differs in exactly: (a)
non-ICL prefill ends at `text[0] ⊕ codec_bos` with no `codec_pad` text block; (b) ICL
prefill overlays the text on the codec prefix positions with the overflow trailing;
(c) the step embedding adds `trailing[k]`/`tts_eos`/`tts_pad` instead of always
`tts_pad` (L1689); (d) nothing changes in RoPE, positions, EOS logic or the prefix-cache
key. Side finding: the official default clone path is streaming, so the C ICL path
currently does not reproduce the official default clone layout; and the C speaker
embedding scaling (`QWEN_SPK_SCALE`) has no official counterpart.

## 4. Chunked prefill taxonomy

| variant | definition | status for this engine | label |
|---|---|---|---|
| Fixed-known-prompt chunked prefill | whole prompt known; KV filled in N-position slices between decode steps (Sarathi-Serve token budget; X-Square Strategy B design) | needs a request-owned cursor, KV written into the slot at `pos = cursor`, per-slice scratch; not present (Codex 2.3) | TRANSFERABLE MECHANISM, DIFFERENT IMPLEMENTATION; safe in principle (pure causal prefix computation) |
| Pause/resume inside a prefill call | return mid-layer/mid-token and continue | no state machine exists; nobody implements it (X-Square runs prefill serially at step boundaries) | unsupported now; not required if slices are whole-position groups |
| Appended/live text | text arrives after generation began | model-supported via the streaming layout: consume one token per frame; park when empty; never pad | DIRECTLY TRANSFERABLE mechanism, quality of pad-then-resume UNKNOWN; protocol details (unfinished tokenizer tail, EOS suppression) from Nari |
| Re-prefill / reconstruction | recompute the prompt (plus a prefix of history) as a new request | already what every span/segment does today; cost = full prefill | supported; solves nothing about admission interference (Codex 2.3 D) |
| Long-form window continuation | segment text; carry state across segments | Talker restart per segment with text tail (X-Square, shipped) is DIRECTLY TRANSFERABLE; decoder-state carry (X2Streaming) DIRECTLY TRANSFERABLE and already structurally present; own-codes re-prompt INSUFFICIENT/negative evidence | see §5 |

Decision logic. With the streaming layout, a preset-voice prefill is ~10-12 positions
(≈50 ms class, per the C1 short-input measurement) regardless of text length, so the
scheduler no longer needs to slice it. What remains long is the ICL clone prefix
(reference frames plus reference text: tens to low hundreds of positions) and any
request kept in non-streaming mode for quality. For those, fixed-prompt chunked prefill
as a budgeted slice (≤ one iteration's reserve, e.g. 32 positions) is the right design;
it is serving-only work (A), while live text (B) rides on the layout and needs the parking
rule, not a prefill state machine. The two must stay separate in PLAN.

## 5. Long-form continuation research

What transfers, in order of evidence strength:

1. **Decoder (Code2Wav) state carry across Talker segments.** X2Streaming carries the full
   Code2Wav bundle (KV, conv and transposed-conv states, frame index) and reports the
   best boundary metrics among chunk-based systems on this exact model. Our per-stream
   `qwen_sd_stream_state_t` already holds transformer KV, latent cache, VQ pad, all
   tails and carries; the engine currently resets it per request/span. Keeping it across
   segments of one logical request is a serving-only change with no model-semantic risk
   (the decoder is causal over codes). Label: DIRECTLY TRANSFERABLE.
2. **Text-tail carry + punctuation-tier segmentation sized by an online EMA of frames
   per token.** Shipped in X-Square; solves the 512-position cap the same way our
   `QWEN_BATCH_MAX_PROMPT` rejection does not. Label: DIRECTLY TRANSFERABLE for the
   windowing rule.
3. **Bounded Talker memory injection (H=4 trailing states, gain ≤ 0.015).** A forward-pass
   modification with a masked memory attention path; inference-only; measured together
   with item 1, so its marginal value is unknown. Label: TRANSFERABLE MECHANISM,
   DIFFERENT IMPLEMENTATION; research arm.
4. **ICL re-prompt with the model's own last-N frames as acoustic prefix.** Expressible
   with the existing ICL path, but the only measured untrained analogue (2603.06444,
   reference dropped) collapsed speaker similarity; VALL-E-continual also lowered SIM.
   Keeping the original reference and appending own frames is untested. Label:
   INSUFFICIENT EVIDENCE; CLI-only falsifier before any server work.
5. MagpieTTS-LF's encoder/prior carry: NOT TRANSFERABLE. Its result that Qwen3-TTS
   whole-text inference reached WER 0.045 on 3-4 minute texts is a useful baseline: the
   model's own long-context handling is not bad; the reason to window is KV/latency
   bounds, not quality.

Bounded quality falsifier for the recommended design (items 1+2): CLI-only, five
long texts (2-4 min), three arms: (a) whole-text non-streaming, (b) segmented with
Talker restart + text tail, decoder state reset per segment (today's span behavior),
(c) segmented with decoder state carried. Metrics: boundary ΔEnergy and ΔF0 at segment
joins, speaker similarity per segment versus the first segment, WER, ear check on joins.
Kill: (c) not better than (b) on ΔEnergy and ear at the joins → decoder carry is not
worth its state handling; (b)/(c) SIM drift > 0.05 across segments → windowing needs
item 3 or a re-supplied speaker prompt.

## 6. vLLM-Omni deep dive (mapped)

| mechanism | problem there | local bottleneck | transfers | does not transfer | falsifier | PLAN impact |
|---|---|---|---|---|---|---|
| Talker/Code2Wav on separate engines + SHM chunks | overlap on two GPUs; host-side step cost | decode burst on the engine thread | the idea of a per-stream frame queue between production and decoding | separate engines, IPC, unbounded deque | F4 (same-pool consumer) | keep AR-1 P4.a as an experiment, not a stage split |
| Producer never throttled (no credit/ack) | none; it is the failure mode (#4913 gaps, unbounded memory) | streams far ahead spend shared bandwidth | the negative lesson | the design | F-C (credit gate A/B) | credit gate in P3 |
| Dynamic initial chunk `[2,4,8,16]` by Talker load; yaml pins 1 | TTFP vs decoder cost per emit | ramp 1,2,2,4,4 then q8 | first chunk = 1 frame always (already) | load-based first chunk: our decoder is stateful, small chunks cost intercept not 10× | F-B (q1/q2/q4 cost) | replace static ramp by lead feedback, §10 |
| AdaptiveChunkController (`buffer_ms = emitted − elapsed`, largest n fitting the lead) | avoid underrun while amortizing decode | same | the signal and the rule | the refusal to shrink (their re-decode cost) | F-B | LS-2 formulation |
| CP re-prefill without KV, batched, fp32 norms | GPU KV machinery cost; numerics drift | CP is L3-bandwidth-bound with 718 barriers | batching across slots (have it); fp32 norms (have it) | statelessness (§9) | none needed | DROP |
| Continuous batching, no batch-forming delay; abort purges before schedule | TTFT; zombie requests | admission FIFO; slot reuse after cancel | both | — | — | keep; OUT-2 cancellation test |
| `input.done` flush as one request | protocol simplicity | our `/v1/tts/stream` is the same | fallback path | — | — | none |
| Code2Wav 300-frame window + 25 left context decoupled from 25-frame delivery | continuity vs transport | our decoder needs no left-context recompute | the decoupling principle (R6) | the recompute | — | principle recorded |

Credit definition for this engine (derived, not copied): per stream,
`lead = delivered_audio − (now − t_first_audio)`;
`credit_frames = clamp(ceil((target_lead − lead) / 80 ms) − frame_queue_len, 0, max_credit)`
where `frame_queue_len` counts produced-but-undecoded frames (lead already committed).
A stream is eligible for a Talker step only with `credit_frames > 0`; a stream with
`credit_frames == 0` and a full frame queue is skipped (its frames would be cache
pollution). Guards: never suppress before first audio; never leave the pool idle while
every stream is suppressed (fall back to stepping the least-lead stream); a parked
live-text stream is simply ineligible.

## 7. Alternative Qwen3-TTS runtimes (implemented vs claimed)

Implemented and read: Nari's one-connection-one-request protocol with `sequence`
numbers, tokenizer unfinished-tail retention, EOS suppression while unfinished, planner
tiers, 1-frame stateful codec steps, drop-and-close backpressure; X-Square's parked
slot, priority tiers, MLFQ, EMA-sized punctuation windows with text-tail carry, bounded
inbox with "Server overloaded" rejection above 256 groups, guarded delivery holding the
last 100 ms to retract hallucinated tails, runtime recomputation of `tts_bos/eos/pad`
embeddings after a 5e-4 export drift flipped CP near-ties; dffdeeq's re-decode-and-crop.

Design-only or roadmap: X-Square Strategy B/C (chunked or dual-engine prefill),
SOFT_DRAIN/COMPACT/RECOVERY, `last_segment_codec_tail`; vLLM-Omni `start_policy`;
Nari has no roadmap markers.

Mechanisms worth adopting (all serving-only): parked-slot semantics for live text;
1-frame first chunk plus lead-driven ramp; drop-and-close output with a byte cap;
overload rejection at admission; EMA frames-per-token for window sizing; an export-drift
check for the special embeddings (cheap parity test against the model file).

Not transferable: FP8, TensorRT fused step, CUDA graphs for the CP loop, paged KV,
re-decode-and-crop windows.

## 8. Scheduling synthesis

Candidates compared on one worker with non-preemptive slices: plain EDF on predicted
underrun (optimal when Σ utilization ≤ 1, domino under overload, no notion of "enough
lead", never lets streams converge into a batch); least-laxity (equals EDF minus a
constant for established streams; the right test only for the prefill job whose
remaining work varies 50-400 ms); credit/DRR alone (proportional degradation, which the
product rule forbids: established streams must not underrun); weighted urgency scores
(no guarantee, re-tuned per ISA). Established results: Liu & Layland 1973; Dertouzos &
Mok 1989; Baruah 1991 and Buttazzo/RED on overload; Shreedhar & Varghese DRR; EEVDF;
Abeni & Buttazzo CBS; Sarathi-Serve; Clockwork; Kairos (URLs in the agent E record).

Recommended primitive: **credit-gated EDF with a reserved prefill budget and
laxity-based admission.**

State per stream: `lead`, `frame_queue_len`, `credit_frames`, `cost_dec` (EWMA per
quantum size), `phase ∈ {prefilling, established, parked}`, and for a prefilling request
`remaining_prefill_positions`, `D_first_audio`. Per worker: `c_T(B)` EWMA per batch
width, `prefill_budget_per_iteration` (≤ ~40 ms so the blocking term stays under one
frame), `U_est`, `panic_lead = 1 frame + max_slice`, `target_lead = 2 frames + p95 step
jitter`, `max_lead = target + 1 frame`.

Decision per iteration: compute leads; E = established streams with `credit_frames > 0`;
if any stream in E has `lead − c_T(|E|) − cost_dec ≤ panic_lead`, run the Talker step for
E now (EDF-forced, no waiting); else if a prefilling request's laxity
`D_first_audio − now − remaining_prefill/rate − c_T − cost_dec` is the tightest and a
prefill slice fits the reserve, run one prefill slice; else run the decode call for the
stream with the earliest underrun that has pending frames; else if E is non-empty and
the batch window (one frame) has elapsed since the first stream became eligible, run the
Talker step for E; else sleep until the next eligibility or panic event. New requests are
eligible immediately with the earliest deadline once their (short) prefill completes, so
first audio is never held for width.

Admission test at arrival: `U_est + ΔU(new at B+1) ≤ U_max` (0.85-0.9), predicted first
audio within `D_first_audio` given the reserve, and no established stream's laxity
below panic with one prefill slice per iteration; any failure → reject (503 with
retry-after), never queue indefinitely. Drift: freeze admission when any laxity < panic
for two iterations; if negative, drop the batch wait (EDF fallback); last resort shed the
newest stream. O(n) scan per iteration, n ≤ 16.

Why lead and not plain EDF: lead adds the "enough" concept (eligibility), which both
bounds bandwidth spent on streams that are far ahead and forms Talker cohorts without
holding first audio; EDF supplies the ordering and the feasibility guarantee among
admitted streams; the reservation isolates prefill; admission keeps Σ U < 1 so the
domino cannot occur. This is one policy with four established components, not a pile of
thresholds.

## 9. CP research verdict: DROP

Stateless re-prefill is DROPPED for this engine. The local audit shows CP on this host
is L3-bandwidth-bound (1.32 GB of INT8 weights streamed per frame per worker, ~86 GB/s,
1-2 MAC per byte) with a secondary barrier term (~4.5 % of busy samples on c8a,
718 barriers per frame in the batched region). Stateless recompute leaves weight bytes
and barriers unchanged and raises MACs 8.1× (136 versus 16 position-forwards per frame);
by the measured row-cost model that is ≈1.7-1.8× the CP projection time, and the algebra
`(16F + 136)/(16F + 16) > 1` holds for every fixed-cost F, so it cannot win at any
bandwidth/compute ratio. It also breaks the region's B ≤ 16 gate at `BW×s` columns.
vLLM-Omni's gain came from removing GPU KV machinery that this engine does not have; its
fp32 numerics practice is already ours (INT8 with FP32 scales and FP32 norms).

Retained from the audit: batched CP across slots at the same codebook index is the only
CP lever and is already implemented (it needs B, hence §12); the CP KV pool allocates 64
slots of which 16 are used (75 % dead memory, trivial fix, no performance claim); the
smallest microbenchmark that would overturn the verdict is a `--rows s` axis on
`tests/region_lowb_bench.c` (predicted ratio ≈1.7-1.8; below 1.0 would contradict the
model), not worth running before P5.

## 10. Initial chunk / steady-state chunk policy

Proposed CPU policy, replacing the static ramp 1, 2, 2, 4, 4, q8:

- First chunk: 1 frame, always (Nari `ttfa` profile; our ramp already does this; vLLM-Omni
  ships `initial_codec_chunk_frames: 1`). Load-based first-chunk selection is not
  adopted: its purpose on GPU is the ~10× left-context recompute of small chunks, which
  our stateful decoder does not pay; our small-call cost is the intercept (23-28 ms at
  q8), which is what P4 attacks.
- Steady state: lead feedback. Next quantum `n = clamp(floor((lead − panic_lead) /
  (cost_per_frame_est)), n_min, n_max)` with `cost_per_frame_est` = EWMA of
  `(intercept + slope×n)/n` for that n, `n_min` = the smallest quantum whose measured
  per-frame cost keeps the worker's utilization under `U_max` (from F-B: expected q2-q4
  today, q1 after intercept reduction), `n_max` = q8 until the fixed-buffer envelope
  shows q12 is affordable. A quantum is never larger than the lead can absorb; a
  stream that is behind target decodes at `n_min`; a stream above target may aggregate
  to `n_max` and may also be skipped for Talker steps (credit gate).
- Load and admission pressure enter only through `U_est` and the prefill reserve, not
  through chunk size: shrinking chunks under load is the vLLM-Omni death spiral in
  reverse and is forbidden by the `n_min` floor.

Smallest falsifier (F-B): q1, q2, q4, q8 arms at C3/C4 on the current binary with the
playback harness; record STREAM p95, required_prebuffer p95, stall_rate@250/500, and the
CT-2 regression per quantum. Kill: no quantum below q8 keeps STREAM p95 ≤ 0.95 at C4 →
the controller has no room until P4 cuts the intercept; the static ramp stays.

## 11. Decoder/preemption correction

External architecture confirms Codex: no runtime yields inside a decoder call. Nari's
"incremental codec" is a stateful decoder invoked per small frame count with persistent
conv histories, transposed-conv overlaps and a sliding pre-transformer KV, i.e. exactly
our `qwen_sd_stream_state_t` (transformer KV and base, latent cache, VQ pad, ConvNeXt
depthwise tails, initial-conv tail, four ConvT carries, twelve residual tails, final
tail, warm flag, arena). The state that would have to survive a mid-call yield (current
stage, intermediate signal buffer, ragged mapping, ConvT carry timing, current output
range, scatter ownership, error/cancel state) is precisely what none of them keep,
because they never yield.

Progression, corrected: Option 1 (smaller complete calls under the lead policy, floor
from F-B) is the P3 mechanism and is what the field does. Option 2 (completed-stage
yield) is throwaway complexity unless the same-pool overlap experiment (Codex 2.5
protocol) shows the consumer needs finer interleave than whole calls; it should not be
built speculatively. Option 3 (strip executor with persistent stage/strip state) is
justified only as intercept and per-frame cost reduction that lets `n_min` fall to 1-2
frames; its state design then coincides with true preemption as a by-product. Ragged
ownership (which items share a call) stays a scheduler decision: aggregate only items
whose leads are both above target (R6).

## 12. Revised architecture implications

- KEEP FROM AR-1: the serialization/coupling diagnosis (as narrowed by Codex 2.1); one
  engine thread per worker with an eligibility/deadline loop; non-blocking bounded
  output; fused residual gated by quality; the do-not-implement list; single engine as
  a later capacity step; the four AMX quantities with unknown whole-request denominator.
- MODIFY: P3 starts with smaller complete decoder calls plus the lead policy, not with
  a resumable decode job; the decode "job state machine" is removed from the P3 critical
  path. Replace plain EDF with credit-gated EDF (§8). Replace the static ramp with the
  lead-feedback quantum (§10). Rename the P3 prefill item: budgeted fixed-prompt slices
  for the residual long prefixes, separated from live text.
- MOVE EARLIER: OUT-1/OUT-2 (drop-and-close, byte cap, send timeout; validated pattern,
  independent, small). The streaming dual-track layout (new SL-1) into P3 ahead of the
  prefill state machine, because it removes the text-length term at lower cost.
- MOVE LATER: fixed-prompt chunked prefill state machine (after SL-1 shows what long
  prefills remain); decoder completed-stage yield (only if F4 demands); strip executor
  stays P4 as intercept work.
- ADD: SL-1 streaming layout for known text (serving-only, model-supported, quality
  falsifier F-A); SL-2 live-text append with parking semantics and Nari-style protocol
  (after SL-1); LF-1 segmented long-form with decoder-state carry and text tail (after
  SL-1, quality falsifier §5); EMA frames-per-token window sizing; special-embedding
  drift check; CP KV pool trim (housekeeping).
- DROP: CP stateless re-prefill; load-based first-chunk selection; own-codes-only ICL
  re-prompt as a plan item (research falsifier only); MagpieTTS-LF mechanisms; stage
  split into separate engines; re-decode-and-crop windows.

## 13. Small research-derived falsifiers (max 8)

| # | hypothesis | experiment | kill criterion |
|---|---|---|---|
| F-A | The streaming layout is quality-equivalent for presets and removes the text-length prefill term | Implement the four layout differences behind a flag (CLI first); same seeds, short/medium/long bank; measure prefill wall and TTFA versus length, mel-corr/WER/SIM versus non-streaming, ear check; one sample compared with the official implementation in streaming mode | prefill wall still grows with length, or WER/SIM/ear worse than non-streaming on presets → keep non-streaming for presets, use streaming for clone only |
| F-B | Small complete decoder calls are affordable under lead control | q1/q2/q4/q8 arms at C3/C4, playback harness, CT-2 regression per quantum | no quantum < q8 keeps STREAM p95 ≤ 0.95 at C4 → intercept work (P4) precedes LS-2 |
| F-C | Credit gating (suppress Talker steps above target lead) improves cadence without RTF loss | emulate on the current binary: frame-queue cap per slot (skip stepping a slot whose undecoded frames ≥ cap) at C4 | prebuffer/stall unchanged and STREAM worse → gating adds nothing at B ≤ 2; keep only the deadline order |
| F-D | Parking (no `tts_pad`) is the correct live-text rule on this model | CLI: generate with a text buffer that empties mid-utterance under (i) park, (ii) pad-then-resume; listen for endings/silences | (i) shows artifacts at resume → live text needs lookahead buffering ≥ k tokens; (ii) acceptable → pad is usable and simpler |
| F-E | Decoder-state carry across Talker segments improves joins | the three-arm CLI test in §5 | (c) not better than (b) on ΔEnergy/ear → carry not worth it |
| F-F | Own-codes ICL re-prompt keeps speaker identity | CLI: window 2 prompted with original reference + last 2-3 s of own codes versus reference only | SIM drop > 0.05 or audible reset on 3 of 5 → drop the idea (matches 2603.06444) |
| F-G | Same-pool decoder consumer overlaps usefully with whole-call decode (Codex 2.5 protocol) | `QWEN_DECODER_THREAD=1`, engine pool, q8, C3/C4, sequential runs | any error/reject/timeout/underrun, material max-gap regression, or STREAM p95 not improved with higher CPU → no overlap in P4 without chunkier calls |
| F-H | Special-embedding export drift exists in the C weights | compare `tts_bos/eos/pad` and `codec_*` control embeddings loaded by the C engine against recomputed values from the model file | drift ≥ 1e-4 relative → fix loader before any CP quality claim |

## 14. Proposed AR-2 delta (exact modifications to the AR-1 P3/P4/P5 proposal)

P3 Cooperative bounded-quantum scheduler and output isolation:
- OUT-1/OUT-2 first (bounded per-stream PCM queue, byte cap, non-blocking writer with
  send timeout, drop-and-close, cancellation; server-side `t_enqueue`/`t_written` per
  chunk to separate inference delay from network delay).
- SL-1 streaming dual-track layout for known text, flag-gated, F-A before default.
- LS-1 per-stream state (lead, frame queue, credit, laxity) + credit-gated EDF loop.
- LS-3' smaller complete decoder calls as the bounded unit (floor from F-B); no
  resumable decode job in P3.
- LS-2 lead-feedback quantum replacing the ramp.
- PF-1 budgeted fixed-prompt prefill slices for residual long prefixes (ICL, optional
  non-streaming presets): request-owned cursor, KV written into the slot; after SL-1.
- LS-4 admission test (utilization + first-audio laxity + reserve), reject over queue.

P4 Overlap and structural decoder cost:
- F-G before any consumer thread; EO-4 only if positive.
- SQ-4 rendezvous/intercept reduction with the counters the audit requires (call sites,
  pool submissions, pool wait time reported separately); fused residual under the audit's
  gate.
- SQ-1' strip body as the intercept project, whose persistent state doubles as
  preemption state.
- Decoder completed-stage yield: only if F-G shows the consumer needs it.

P5 Single engine and batching economics: unchanged in substance (EO-1/EO-2/EO-3, AMX at
B ≥ 4), with the cohort formed by the credit gate rather than by waiting; F6 first.

Research arms (gated, CLI first): SL-2 live text with parking; LF-1 segmented long-form
with decoder carry and text tail (F-E); F-F own-codes re-prompt; Talker memory injection
(X2Streaming H=4) only after LF-1; PREFILL-Q unchanged, deferred.

## Conclusion

External evidence does not overturn AR-1's serving diagnosis; it re-ranks the remedies.
The largest missed lever is the model's own streaming layout, which makes first play
independent of text length without a prefill state machine. The field bounds decode
bursts with small complete stateful calls, which matches the Codex correction and puts
intercept reduction, not resumability, on the decoder critical path. The scheduling
currency is audio lead with an eligibility gate and EDF ordering, the pattern vLLM-Omni
lacks and pays for. CP statelessness and untrained own-codes continuation are closed.

## Next action

AR-2: apply §14 to PLAN; run F-A (CLI, after a flag-gated layout implementation), F-B,
F-C and F-G on the current binary as the first Tier A session; keep F-D/E/F/H as
CLI-level research falsifiers that gate SL-2 and LF-1.

## Explicit answers

1. P3 begins with smaller safe complete decoder calls plus lead scheduling; true decoder
   resumability first is not justified (no runtime does it; Codex proved the state does
   not exist; the strip's state design will provide it later as a by-product).
2. Fixed-prompt chunked prefill enters P3 only as a budgeted slice for the residual long
   prefixes, after SL-1; not as the first P3 item.
3. True incremental/live text is technically justified for this model (trained
   dual-track, official streaming layout, two source-level runtimes), staged as SL-1
   (known text, serving-only) then SL-2 (live append with parking); the pad-then-resume
   case stays research (F-D).
4. CP stateless re-prefill: dropped (bandwidth-bound, bytes unchanged, MACs ×8, barriers
   unchanged); keep batched CP for P5.
5. Yes: audio lead with a credit/eligibility gate replaces plain EDF as the central
   currency; EDF remains the ordering rule inside the eligible set.
6. First chunk stays static at 1 frame; the steady-state quantum becomes lead-adaptive
   with a utilization floor; load-based first-chunk selection is not adopted.
7. Yes: output isolation and the streaming layout move ahead; the decode job state
   machine and the prefill state machine move later; overlap stays gated by F-G; single
   engine unchanged.
8. Yes: the official streaming dual-track layout (text consumed one token per frame,
   prefill independent of text length, park-not-pad semantics) was missed by both AR-1
   and the Codex audit; secondarily, decoder-state carry across segments is already
   structurally present and externally validated on this model.
