# SL-1 known-text dual-track streaming layout

Date: 2026-09-07
Implementation baseline: 2a8b0412cec37b095b456c5b125b7beb49190cd8
Status: implemented, default-off, local gates passed; server/quality promotion pending.

## Scope

QWEN_TTS_STREAM_LAYOUT=1 implements the official known-text
non_streaming_mode=False schedule. It does not accept live network text and does
not make the existing prefix cache a resumable prefill cursor.

The implementation keeps the current non-streaming path unchanged when the flag is
unset or 0. It is carried through:

- single-request CLI generation;
- CLI --stream;
- qwen_tts_generate_batch;
- qwen_tts_generate_batch_multi;
- continuous-server admission, including the optional prefill helper.

Trailing text vectors are request-owned. The context owns them until a batch or
server slot/prefill result explicitly transfers ownership; those owners free them
on normal completion, rejection, cancellation/error cleanup and unload.

## Official-to-C mapping

For non-ICL known text, the prefill now ends at the unchanged role/control prefix
plus one aligned position containing the first text token and codec BOS. The
remaining text token embeddings and one tts_eos embedding are prepared once and
consumed one per generated Talker step; after exhaustion the step uses tts_pad.
Generated codec EOS remains the stopping token. KV positions remain monotonic and
there is no RoPE reset.

For ICL, the common prefix overlays reference text/request text/tts_eos with
codec BOS/reference codec frames. Any unpaired text tail is retained as trailing
text; text is padded with tts_pad when the codec track is longer. Control,
speaker/instruct, graft and local EOS policies are unchanged.

The semantic source audit is in
.work/ar2-sl1-semantics-20260907.md.

## Local validation

Passed:

- make clean && make blas
- ./qwen_tts --caps
- ./qwen_tts --self-test (0 failures)
- python3 tools/check_flag_registry.py
- python3 tools/flag_parity.py --check
- bash -n tests/stream_layout_smoke.sh
- tests/stream_layout_smoke.sh

The smoke covers short/medium/long non-ICL text with deterministic greedy
generation. It observed:

| input | stream common positions | trailing text positions | output frame count |
|---|---:|---:|---:|
| short | 1 | 4 | equal control |
| medium | 1 | 15 | equal control |
| long | 1 | 31 | equal control |

The feature also completed a direct CLI --stream smoke. Local ICL/clone assets
are unavailable: the checked-in base-model links point to an unavailable external
volume, so the optional ICL test is explicitly skipped rather than claimed as
covered.

make test-golden remains blocked by the pre-existing local librosa/Numba cache
failure in tests/compare_audio.py; this is an environment failure, not a reported
audio comparison result. The new smoke still verifies valid WAV headers, non-empty
output and equal generated frame counts. The streaming schedule is a model-visible
change, so audio quality and upstream-layout comparison remain required before
promotion.

The local server bind smoke could not run in the managed macOS sandbox
(bind: Operation not permitted). Actual continuous-server batching validation is
still required on the canonical GCP AMX host before this flag can be promoted.

## Current decision

PROMOTE implementation behind the default-off flag for CLI and server integration
testing. KEEP default-off pending:

- GCP continuous-server/batched-path census and reuse evidence;
- ICL/clone validation where the base model and reference assets are available;
- established audio-quality/quality-bank gates;
- prefill and TTFA scaling evidence versus the unchanged non-streaming control.

Live incremental text, append-after-generation, fixed-prompt resumable prefill and
output/scheduler redesign remain separate P3 work.
