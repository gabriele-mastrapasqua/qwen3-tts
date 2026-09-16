# Post-P2 streaming architecture research agenda

Task: AR-1 AR-2 (post-P2 architecture review and the P3/P4 re-ordering it produces)

## Question

After the small-quantum decoder (P2) lands, what should THIS engine become so that
first play is bounded independently of input length, delivery stays continuous at 1x,
realtime margin and fairness hold under concurrency, and the result is stable across
utterance lengths? Which of the ideas below survive contact with the actual Qwen3-TTS C
dataflow, and in which order should Codex implement them?

## Known facts

- Objective (PLAN Mission, `.work/professional-streaming-architecture.md`): bounded
  startup + continuous playable cadence + realtime margin + fairness under concurrency +
  stability across input length. Hardware-efficient compute is subordinate to that
  envelope. Aggregate STREAM_RTF and TTFA alone are insufficient (E1, E2).
- P1 (`.work/p1-cadence-truth-20260907.md`): the cadence law is confirmed (q32 required
  prebuffer p95 2.2-2.5 s versus q8 0.3-0.7 s at C4 while RTF moves less); C2/C3 GOOD,
  C4 MARGINAL on the current architecture; inline admission prefill of a long request is
  about 309 ms and coincided with a roughly 235 ms larger established-stream gap (one
  pair, not a causal model); Talker B2/B1 step ratio about 1.10 (EO-2 retained).
- Admission and prefill are inline on the engine thread (`ADMIT_PREFILL`), the decoder
  is inline and gang-joined, output is three blocking writes per chunk (E5, E12).
- P2 (SQ-1/SQ-2/SQ-3) is being implemented by Codex; its evidence is not yet complete
  and must not be reinterpreted here.

## Unknowns

- Whether the current engine's TTFA grows with input length through prefill and
  admission (R1), and how much of that reaches established streams.
- Whether Qwen3-TTS state permits incremental or preemptible prefill without prosody
  resets (R2), and what recomputation a text-window boundary really costs.
- Whether bounded-window generation with explicit continuity state is achievable on this
  model (R3), and whether history cost defeats the startup gain.
- Which vLLM-Omni staging mechanisms survive translation to a C CPU engine with shared
  LLC/DRAM and a bounded core count (R4).

## Dependency and blocking rule

**Do not execute this research or the review until Codex has completed P2, committed its
runtime work and evidence, and produced a stable checkpoint.** A separate read-only
architecture review will then be requested. That review MUST audit the new P2 HEAD and
its evidence; nothing here may be reasoned from the pre-P2 decoder. Until then this note
is a preserved agenda, not a work order, and P3 does not start on the basis of it.

## Files/functions inspected

`PLAN.md`, `ENGINEERING.md`, `.work/professional-streaming-architecture.md`,
`.work/p1-cadence-truth-20260907.md` (this note records questions; no code was
inspected for it beyond what those addenda already anchor).

## Evidence

None new. This note is an agenda. The evidence it needs is produced by the future
review; each item below names the falsifier that review must run.

## Agenda

### R1 — Input-length-independent first play (high priority)

Principle: first-play latency should be bounded primarily by the initial synthesis
window, not by total utterance length. Historical observation to investigate, not a
current benchmark fact: short inputs reached roughly 200 ms-class TTFA while much longer
inputs rose toward 700 ms or more on earlier backends/configurations. After P2, determine
whether the current engine exhibits the chain input length → larger prefill → larger
admission stall → larger TTFA → interference with established streams. Required
evidence, per request and separated: input token/text length, prefill wall, admission
wait, first generation step, TTFA, established-stream cadence impact during the event,
total output duration. Naive sentence splitting is not an accepted solution.

### R2 — Incremental / chunked / preemptible prefill

Research whether the engine can move from full input → monolithic prefill → generation
toward bounded initial text window → initial prefill → generation starts, with later
text-window preparation in parallel. Questions: which Qwen3-TTS state makes incremental
prefill possible or impossible; which KV/text-conditioning state can be retained; whether
later windows can be prepared without resetting prosody; whether prefill can be a
bounded/preemptible scheduler job; whether established playback deadlines can defer
later prefill; whether later-window prefill can overlap codec generation or decoder work;
what recomputation a window boundary really requires; what model-semantic change, if
any, is needed. Separate clearly a serving-only change from a model/inference-algorithm
change. Do not assume incremental prefill is possible until the model dataflow proves it.

### R3 — Long-form stateful generation and continuity

Study, extract mechanisms, then map them against the actual Qwen3-TTS implementation
(none is assumed to transfer): MagpieTTS-LF (arXiv 2606.18485), the history-prompting
mechanism in arXiv 2412.18603, VALL-E (arXiv 2301.02111), and Bark's long-form
`history_prompt` notebook. Questions: can text be divided into bounded synthesis windows
without audible resets; what generated codec-code tail can be retained or re-prompted;
what text tail must accompany it; do speaker identity, accent, energy and prosody stay
stable; how much history is sufficient; does history raise CP/Talker cost enough to
defeat the startup benefit; how are pauses and sentence boundaries handled; does rolling
context accumulate drift; how is deterministic continuity validated; can a bounded
rolling history keep context from growing forever. The target is bounded-window
generation with explicit cross-window continuity state, not concatenated WAVs.

### R4 — Async staged TTS serving (vLLM-Omni)

Study the paper (arXiv 2602.02204) and repository (vllm-project/vllm-omni) for
transferable scheduling/dataflow mechanisms, not for the runtime: the Qwen3-TTS serving
path, async chunk mechanisms, stage ownership, inter-stage queues/connectors, downstream
decoder scheduling, per-stage batching, backpressure, dynamic chunk sizing, admission,
cancellation, output delivery, incremental/streaming input support. Ask which mechanisms
survive when translated to a C CPU engine with shared LLC/DRAM, AMX/VNNI and a bounded
number of physical cores. Accelerator-oriented structure is not assumed appropriate.

### R5 — Decoupled pipeline

Evaluate whether the target becomes: request/admission → bounded or incremental prefill →
global Talker/CP ready scheduler → bounded per-stream codec-frame queue → sliced or
decoupled decoder executor → bounded per-stream PCM queue → non-blocking output, with
playback-lead/deadline feedback into scheduling. For each stage define ownership,
mutable state, queue boundary and bound, backpressure, scheduling deadline, batching
opportunity, AMX/VNNI opportunity, memory-bandwidth consequence, cancellation, failure
isolation, expected latency contribution. Do not recommend separate thread teams merely
because stages are conceptually separate; compare with CPU evidence at least (A) same-pool
bounded slices, (B) a dedicated lightweight stage/lane, (C) a fully separated executor.

### R6 — Granularity decoupling (core principle)

Hardware compute granularity must not dictate client delivery granularity. The server
may aggregate work internally for efficient matrix execution while delivering small,
regular PCM increments. Never repeat "larger decoder chunk → better amortized RTF →
worse cadence". Investigate how bounded queues, decoder slicing, lead-aware scheduling,
internal batching, persistent packed weights and cache-resident strips together permit
hardware-efficient work and low-jitter delivery.

### R7 — Playback lead as the cross-stage currency

Extend the lead scheduler (LS-1/LS-2) to admission, later-window prefill, Talker batching
delay, decoder scheduling and opportunistic preparation work: very low lead → generation
and decode deadlines dominate; moderate lead → small batching allowed; high lead → spend
slack on future text/prefill work. Prefill becomes a deadline-aware job rather than an
unconditional inline stall. The review must name the starvation and fairness failure
modes; new requests keep a bounded first-audio deadline.

### R8 — Long-input and concurrency qualification

The professional benchmark must not qualify only short prompts. Add dimensions: short,
medium, long and mixed inputs; established streams + arrival of a long request; long
request + arrival of a short request; stationary workload; slow reader;
cancellation/barge-in; sustained soak. Each retains the full envelope: TTFB, TTFA,
STREAM_RTF and margin, required_prebuffer, safe_play_start, max_gap, underrun,
stall_rate@100/250/500/1000, errors/rejects/timeouts, fairness. Capacity = maximum
sustainable gap-free streams under the complete envelope, not accepted concurrency.

### R9 — PREFILL-Q: prefill precision / INT8 / AMX (deferred, below architecture)

Prefill is one of the more matrix-friendly parts of the model, but earlier simple INT8
prefill experiments were not production-qualified and could alter output quality. Do not
reopen simple INT8 prefill as an immediate P3 optimization. Deferred arm: can a
quality-oriented, calibration/rounding-aware quantization give a safe prefill speedup
without the quality/accent/prosody degradation seen before? No method is adopted from
benchmark claims. Gate: same serving architecture, same prompts/seeds, rigorous
audio/semantic quality comparison, then latency/throughput. A faster monolithic prefill
is useful; a bounded/preemptible/incremental prefill changes the serving problem more
fundamentally, so architecture first.

### R10 — Modern x86 / AMX utilization after the dataflow is right

This redesign does not abandon AMX. After the streaming dataflow is established, ask
where modern x86 matrix hardware can be exploited without harming cadence: Talker
batching, prefill matrix work, decoder Design D, the strip executor, future stage-local
batching. Keep the four separate concepts (amx_dispatch_share, amx_matrix_mac_share,
amx_addressable_mac_share, amx_request_wall_share). The objective is less wall time per
useful unit of continuous playable audio, never more AMX calls.

## Future review deliverable (when requested after the P2 checkpoint)

1. Audit the committed P2 HEAD and evidence. 2. Identify what remains inline/blocking.
3. Research the long-form/stateful literature above. 4. Inspect the serving-runtime
code/paper above. 5. Map mechanisms to the actual Qwen3-TTS C engine. 6. Separate
serving architecture, inference/model algorithm, kernel optimization. 7. Propose ONE
coherent target architecture. 8. Rank implementation phases. 9. Define a cheap falsifier
for every major claim. 10. Name the ideas that should NOT be implemented. The output must
answer "what should this engine become", implementation-oriented for Codex, not a survey.

## Conclusion

No decision is taken here. The agenda is preserved so that the post-P2 review reasons
end-to-end (startup, cadence, margin, fairness, length stability) instead of optimizing
one kernel or one latency number, and so that P3/P4 ordering is frozen from that review
rather than continued blindly.

## Next action

Wait for the Codex P2 checkpoint (runtime committed, evidence addendum, stable HEAD).
Then request the read-only architecture review (AR-1) against that HEAD, and only after
it freeze the revised P3/P4 ordering (AR-2) before Codex resumes implementation.
