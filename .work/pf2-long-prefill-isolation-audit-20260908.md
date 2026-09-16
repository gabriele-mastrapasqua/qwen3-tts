# PF-2 long-prefill isolation audit — 2026-09-08

Task · Establish the causal long-input path on the frozen 8-core AMX serving
reference and decide whether the already implemented SL-1 known-text layout is
the first bounded falsifier.

Question · Can an accepted long request be admitted without consuming the
playback safety budget of established streams, and is the current runtime state
ready for cooperative prefill?

Known facts · The current 1.7B reference is `1x8`, cap 3, `--max-queue 0`, q4,
Design-D INT8 decoder, fused residual, warm strip, ragged threshold 2,
engine-owned pool and synchronous output. The full C3 envelope is not qualified:
long startup and accepted long-prefill admission are the remaining material
continuity failures. The 2+1 probe measured injection-window max gaps of
approximately 525–594 ms and fixed-buffer stalls at 250 ms in two of three
repetitions.

Unknowns · SL-1 has a bounded non-ICL known-text prefill-scaling result and a
short server path gate, but no current 8-core long-arrival A/B. It is a
model-visible layout and therefore needs same-generation structural/audio gates.
No request-owned resumable Talker prefill state exists at this checkpoint.

Files/functions inspected · `qwen_tts.c` (`qwen_tts_generate`,
`qwen_tts_generate_batch_multi`, `ADMIT_PREFILL`, `ADMIT_INSTALL`),
`qwen_tts_talker.c` (`qwen_talker_prefill`), `qwen_tts_server.c` (parent
admission, continuous scheduler and prefill helper), `qwen_tts.h`, the SL-1,
input-length, prefill-helper, LS-1, LS-4, F1, fused-residual, admission and
serving-operation evidence.

## Current causal map

The normal prefork parent accepts a connection, accounts for a worker's whole
connection slot and sends the descriptor. The child scheduler then calls
`qwen_tts_serve_continuous`. When a slot is free, `sink_next_job` returns a
request and the inline `ADMIT_PREFILL` path performs `qwen_tts_generate` with
`prefill_only=1` before `ADMIT_INSTALL` copies KV and final hidden state into the
batch slot.

For the ordinary preset Ryan/English request, this path is non-ICL known text:
the request has no reference audio/text, no voice-clone state and no custom
instruct state in the qualification bank. The full-layout path tokenizes the
whole text, creates the full prompt and calls `qwen_talker_prefill` once. The
prefill uses the resolved native BF16/matmat path on the AMX product profile
where its shape gate applies; it is not the decoder's Design-D INT8 path.

The scheduler loop is the owner of the admission transition. Prefill projection,
attention and feed-forward work use the engine-owned compute budget/pool, but
the scheduler does not regain control between the request's prefill and
`ADMIT_INSTALL`. Consequently established Talker/CP/decoder progress is
structurally blocked by the inline call and also contends for the same compute
team while that call is active. This is stronger than mere idle-core
underutilization: the loop has no opportunity to select an established frame
until the admission call returns. Average core-equivalent utilization therefore
cannot prove safety; the long-prefill probe directly observed cadence damage.

The prefix cache can reuse a matching request-independent prompt head when its
key and cache entry match. It does not turn the variable text suffix into a
resume cursor, and the server explicitly resets `prev_prefill_len` before each
admission. The variable text still participates in the full prefill. The
current per-request handoff copies KV and final hidden state only after the full
prefill has completed.

## SL-1 applicability

`QWEN_TTS_STREAM_LAYOUT=1` is read by `qwen_tts_generate`, so the continuous
server admission path and the batch path use the same implementation. For
non-ICL known text it keeps the role/control prefix plus the first text token
and codec BOS in the initial prompt. Remaining text embeddings and `tts_eos`
are request-owned trailing vectors consumed alongside subsequent generated codec
frames. KV positions stay monotonic; generated codec EOS remains the stopping
condition. It does not accept live network text and does not pause/resume a
prefill.

The frozen AMX product profile explicitly sets this flag to `0`; a treatment
can therefore change only this variable. Ryan/English server requests are
compatible with the non-ICL branch. ICL/clone, emotion-reference and custom
voice paths use the separate official alignment logic and are not silently
covered by the non-ICL result. The current campaign should begin with the
known-text Ryan/English workload and retain the existing structural/audio
qualification caveat.

SL-1 reduces initial prefill positions but adds one hidden-vector/text-tail
addition to generation steps. It can flatten the input-length-dependent
startup term while changing steady-state cost and model-visible scheduling.
Only an A/B with identical server settings can determine whether it protects
established streams.

## State-boundary audit

| mechanism | classification | current conclusion |
|---|---|---|
| SL-1 known-text dual-track layout | A — immediately implementable | Already implemented and default-off; safe first A/B for non-ICL known text. |
| Existing prefix cache | D — already insufficient | Reuses a fixed prompt head only; not resumable variable-text prefill. |
| Existing `prev_input_embeds`/`prev_prefill_len` | D — already insufficient | Context-local comparison/cache aid; server admission resets the cursor and still runs a full prefill. |
| Full prefill then KV/final-hidden handoff | D — current control | Correct and bounded, but cannot yield during the call. |
| Request-owned layer/token prefill cursor | B — bounded new state, if boundary is proven | Would need ownership of residual, Q/K/V/attention/FFN scratch, layer/token progress and KV output. No current interface exposes it. |
| Transformer-layer yield with the current vectorized prefill | B/C — requires restructuring | A layer boundary is conceptually safer than a GEMM-block boundary, but current scratch/state is context-owned and the scheduler/pool contract has no continuation object. |
| Token-block continuation through all layers | C — architectural | Causal attention and per-layer residual/KV dependencies require explicit state and a new execution schedule; recomputing the whole prefix is not acceptable. |
| `qwen_talker_step` per text token as fake prefill slicing | D — reject as a substitute | It would change the arithmetic path/cost and may alter numerical/order semantics; it is not resumable vectorized prefill. |
| Existing prefill helper + LOW priority | D — experimentally rejected | It still computes one complete prefill and added multi-second TTFA tails without protecting serving cadence. |
| Same-pool decoder consumer / static lanes | D — rejected for this generation | Existing evidence showed severe C4 tail/cadence regression; not a PF-2 shortcut. |

## Decision before implementation

The long-prefill root cause is established as inline full-prompt Talker prefill
inside the worker scheduler, with same-pool resource contention. The smallest
justified next action is the SL-1 control/treatment falsifier on the known-text
Ryan/English workload. It changes no topology, quantum, admission, decoder,
pool or output policy.

True cooperative prefill is not yet an immediately safe code change. A valid
PF-2 implementation would require request-owned continuation state and an
explicit safe boundary; it must not be fabricated by restarting or recomputing
the complete prefix. If SL-1 fails to reduce both long startup and accepted
long-arrival disturbance, the next result should be a bounded-state design
decision, not more helper/LOW or arbitrary millisecond tuning.

Next action · Run the fixed SL-1 A/B: C1 long, isolated C3 long, accepted 2+1
long-arrival, and one short/medium regression, with the frozen 8-core AMX
profile otherwise unchanged. Do not begin PF-2 implementation before that
falsifier is complete.
