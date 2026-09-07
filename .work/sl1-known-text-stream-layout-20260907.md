# SL-1 known-text dual-track streaming layout

Date: 2026-09-07
Implementation baseline: f490d65e81d3603e3b05960fc42041a369524250
Status: implemented, default-off, local and GCP Tier-A path gates passed; quality
promotion remains pending.

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
(bind: Operation not permitted). This was covered by a short continuous-server
run on the GCP AMX host below.

## GCP Tier-A validation

Host state was checked immediately before the runs:

- `c4-standard-24`, Intel Xeon Platinum 8581C, one socket/NUMA;
- 12 physical cores, online `0-11`, `Thread(s) per core: 1`;
- `/sys/devices/system/cpu/smt/control=off`, `active=0`;
- clean source snapshot identified locally as `f490d65`; the remote archive has
  no `.git`, so the harness's remote `source_commit=unknown` is not evidence of
  a different source. The binary SHA-256 prefix was `db5b79e6eaed2929`.

The remote build was clean (`make blas SIMD=amx` after `make clean`) and passed
`--caps` and `--self-test`. The server arms used the same P2 controls:
`QWEN_SD_AMX_D=1`, `QWEN_SD_STREAM_STRIP=1`, `QWEN_SD_RAG_MIN_PANELS=2`,
`QWEN_STREAM_DECODE_CHUNK=8`, `QWEN_SD_POOL=engine`, `QWEN_DECODER_BATCH=1`,
2x6 prefork, batch cap 2. The only A/B variable was
`QWEN_TTS_STREAM_LAYOUT=1`.

Short one-wave true-simultaneous waves used the existing 1.7B English bank;
they are a path/integration gate, not a qualification campaign:

| arm | C | TTFA p50/p95 (ms) | STREAM_RTF p50/p95 | required prebuffer p50/p95 (ms) | safe start p50/p95 (ms) | max gap p95 (ms) | fixed stall @500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 99/99 | 0.515/0.515 | 12/12 | 111/111 | 329 | 0% | 0/0 |
| SL-1 | 1 | 78/78 | 0.511/0.511 | 13/13 | 91/91 | 327 | 0% | 0/0 |
| control | 2 | 105/117 | 0.550/0.577 | 25/28 | 130/145 | 382 | 0% | 0/0 |
| SL-1 | 2 | 82/90 | 0.536/0.589 | 22/28 | 104/118 | 372 | 0% | 0/0 |

Receive fidelity was acceptable for this small gate: coalesced-read share was
`6.7%/7.4%` for control C1/C2 and `7.1%/8.7%` for SL-1 C1/C2. The result is
not a server qualification result: one wave, short bank and client-observed
cadence remain insufficient to establish steady-state quality or margin.

A separate SL-1 C2 census run, with the same topology and flags plus
`QWEN_SHAPE_CENSUS=1`, recorded 3 census files, 586 frames, 133 rows, zero
dropped operations and zero unknown/fallback calls. The executed decoder rows
were `AMX` Design-D (`decoder_conv_amx_int8_design_d`) plus the existing BLAS
decoder work: decoder GMAC classification was 85.2% AMX/optimized and 14.8%
BLAS, with 0% fallback. This proves the feature ran through the real batched
server while the existing AMX decoder path remained engaged; census timing was
not used as a KPI.

The run did not have the official Python model/runtime or ICL reference audio
installed, so it cannot establish upstream audio equivalence for clone/ICL.
The control-vs-SL-1 waveform is not a semantic oracle: the prompt layout is
intentionally model-visible. Therefore SL-1 remains default-off pending an
upstream-layout quality gate, ICL/clone coverage and text-length prefill/TTFA
scaling evidence.

## Current decision

PROMOTE implementation behind the default-off flag for CLI and server integration
testing. KEEP default-off pending:

- GCP continuous-server/batched-path census and reuse evidence (now PASS for the
  Tier-A path gate above; broader serving qualification is still pending);
- ICL/clone validation where the base model and reference assets are available;
- established audio-quality/quality-bank gates;
- prefill and TTFA scaling evidence versus the unchanged non-streaming control.

Live incremental text, append-after-generation, fixed-prompt resumable prefill and
output/scheduler redesign remain separate P3 work.
