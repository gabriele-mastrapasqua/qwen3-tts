# Server-vs-CLI Italian correctness — independent review (NOT the execution track)

**Task.** Prepare an interpretation framework for a reported server-only Italian
pronunciation defect, by inspecting the current code for places where the server path and
the CLI path differ SEMANTICALLY. **Question.** Where is the first point at which a request
served through the batched server stops being the same computation as the same request run
on the CLI? **Known facts.** CLI pronunciation of the reproducible Italian case is good;
server streaming pronunciation is wrong; English controls appear good; ASR agrees with the
listener on at least some bad Italian outputs; reproduced on both Turin and a Mac.
**Unknowns.** Everything causal. No experiment in this document has been run by its author.

## 0. OWNERSHIP — read this first

The live forensic on this defect is owned and executed by a separate session. **This
document does not execute anything and must not be used to start a parallel investigation.**
It exists to make the returning evidence faster to interpret, and to record three
CLI-vs-server asymmetries found by reading the code that any forensic should be aware of
before it starts instrumenting.

Two constraints that follow from ownership: do not modify instrumentation the execution
owner may be relying on, and do not treat any partial result as a conclusion.

## 0.1 ROOT CAUSE FOUND BY THE EXECUTION TRACK (2026-09-10) — and what it costs us

The live forensic located the first divergence in the server's TEXT INPUT, before prefill,
KV or CP: `json_extract_string()` in `qwen_tts_server.c` performs **no unescaping at all**.
It uses the backslash only to skip a character while scanning for the closing quote, then
`memcpy`s the raw bytes. Verified by reading the function.

So the defect is broader than the reported `\uXXXX` case. Every JSON escape reaches the
tokenizer literally: `\uXXXX`, `\"`, `\n`, `\\`, `\t`, `\/`. A request containing a quoted
phrase or a newline is corrupted in ANY language, English included; the language pattern in
the symptom comes from WHICH clients emit escapes, not from the language itself.

Why CLI is clean and server is not: the CLI takes `--text` from `argv` as raw UTF-8 and
never passes through this parser.

**This is a product interoperability defect, not a harness artifact.** Python's
`json.dumps` defaults to `ensure_ascii=True`, as does much other tooling, so a standard
client sends `\uXXXX` for every non-ASCII character. That breaks every accented language and
destroys CJK entirely (each character becomes a six-character literal). Sending
`ensure_ascii=False` is the correct isolated CONFIRMATION and it fixes our harnesses; it is
NOT the fix. The fix belongs in the parser. No repository harness that sends text currently
passes `ensure_ascii=False`.

### Evidence integrity — what survives and what does not

Every serving campaign that sent non-ASCII text through the JSON API fed the engine text
that was not the text we intended.

* **Survives**: paired A/B comparisons where BOTH arms went through the same corrupted
  path, because both arms received identical (corrupted) input. That includes the RES1_V2
  reference-vs-V2 parity result and the spec 10 server parity harness gates.
* **Does NOT survive**: any ABSOLUTE Italian quality claim, any CER/WER scored against the
  intended text, and any statement that an Italian bank "sounded right". The V2 quality
  qualification keeps its parity meaning and loses its semantic-quality meaning.
* **Marginal**: performance numbers. The corrupted Italian prompts carried a few extra text
  tokens, so the short/medium/long class boundaries were slightly off for Italian; RTF
  conclusions are not materially affected.

### Why the frame-count difference follows

More text tokens raise `ctx->bg_text_content_len`, which the server captures as `tcl[b]` and
feeds to its hardcoded EOS heuristic (section 3.1). The 53-versus-55 frame difference is a
CONSEQUENCE of the corrupted input, not an independent stopping-policy bug. Section 3.1
remains a real CLI/server asymmetry, but it is not the cause here and must not be "fixed"
on the strength of this case.

## 1. What must NOT be assumed

* **Not spec 10.** `QWEN_PREFILL_SLICE` is default OFF and this defect predates it. It can
  only be implicated if the flag is actually enabled in the failing run.
* **Not RES1_V2**, merely because it is recent.
* **Not batching**, until B_eff=1 evidence exists.
* **Not "Italian is special"** — CONFIRMED in section 0.1: the trigger is which characters
  a client escapes, not the language. English is exposed too, through `\"` and `\n`.

## 2. Four categories that must never be collapsed

Today's spec 10 work is the methodological lesson, not evidence about this defect. It showed
that paths which look equivalent can cross different execution boundaries (f32-resident vs
bf16-cached state; M=1 matvec vs M>1 matmat), and that a difference far below a bf16 ulp
flips a discrete choice and changes an entire utterance. So classify evidence as:

1. **SEMANTIC STATE DIFFERENCE** — wrong language id, position, KV length, prompt length,
   token history, stopping policy. The state itself is not the same.
2. **NUMERICAL PATH DIFFERENCE** — the same state mathematically, different kernel,
   precision or accumulation order.
3. **DISCRETE GENERATION DIVERGENCE** — the first codec/token choice that actually flips.
4. **DOWNSTREAM AUDIO DIFFERENCE** — codec tokens identical, waveform differs.

A small max_abs/RMS in category 2 is NOT harmless when an autoregressive discrete choice
sits downstream. Report the first index at which category 3 occurs, not only the magnitude.

## 3. Asymmetries found by reading the code (facts, not hypotheses)

These are differences that exist in the source today. Whether any of them causes the symptom
is unproven.

### 3.1 The stopping policy is not the same function

* CLI: a configurable EOS strategy — `eos_strategy` V1/V2, `eos_overhead_frames` (18),
  `eos_frames_per_token` (3.0), `eos_suppress_frames`, `eos_start_multiple`,
  `eos_ramp_per_frame`, `eos_ramp_cap` (qwen_tts.c around the generate loop).
* Batched server: `SAMPLE_SLOT` hardcodes `_ef = tcl[b] * 3; _bs = _ef * 2;` then a
  `0.5 * (sframe - _bs)` EOS bonus capped at 10.

So the server applies a fixed length heuristic where the CLI applies a parameterised one,
and the server's input is `tcl[b] = ctx->bg_text_content_len` captured at admission. A
length-dependent stopping policy that differs between the two paths would show up as
**endings clipped or words eaten**, and would look language-dependent purely because
Italian and English tokenize the same content to different lengths. Category 1.

### 3.2 The waveform decoder runs a different shape of computation

* CLI (default, writes a WAV): `qwen_speech_decoder_decode(ctx, chcodes, chframes, ...)` —
  the WHOLE code sequence in one call.
* Server streaming: `qwen_speech_decoder_decode_streaming_st(ctx, &sstate[b], new_codes,
  new_frames, ...)` — CHUNKS, with per-slot causal state carried across calls, optionally on
  the decoder lane.

This is the largest structural difference downstream of the tokens. A causal-history error
at chunk boundaries produces exactly "locally merged / eaten / transformed" audio while the
codec tokens remain identical — category 4. Italian would be perceptually more sensitive to
it (geminates, final accented vowels, elisions) without the cause being language-specific.

### 3.3 Language sets the speaker on one path only

`qwen_tts_set_language` overrides `speaker_id` for English/Chinese/Japanese/Korean (e.g. EN
forces 3061). The batched server assigns `ctx->language_id = req.language_id` directly and
never routes through that helper. For Italian neither path overrides, so this does not
explain an Italian-only symptom — but it does mean the two paths are not the same function
of `(speaker, language)`, and it is a trap when constructing "identical" CLI and server
control runs. Category 1, relevant to experiment design rather than to the defect.

## 3.4 MEASURED 2026-09-10 — the streaming decoder is NOT the mechanism

Run entirely on a dev machine, no server, no cloud, and deliberately NOT a duplicate of the
live forensic: this compares the CLI against ITSELF, changing only the shape of the decode.
Same model (1.7B CustomVoice, ryan), same Italian sentence in the reproducible class, same
`-T 0 --seed 42 --int8`, so the generation is identical and only the waveform decoder
differs — whole-sequence `qwen_speech_decoder_decode` versus chunked
`qwen_speech_decoder_decode_streaming_st` at `--stream-chunk 4` (the server's quantum) and
`--stream-chunk 10`.

| comparison | codec tokens | audio |
|---|---|---|
| whole vs chunk 4 | **bit-identical** (73 frames, same sha) | mel-corr 0.99975, rms 9.7 LSB on a 15607 peak, 0.7 % of samples differ |
| whole vs chunk 10 | **bit-identical** | same magnitude |
| chunk 4 vs chunk 10 | bit-identical | max 1 LSB over 24 samples — chunk-size invariant |

The only frame exceeding 2 % of peak error is the LAST one (frame 72 of 73), i.e. the
end-of-utterance tail flush, not a per-chunk-boundary effect — and chunk-size invariance
confirms that reading: if chunk boundaries were corrupting state, quantum 4 and quantum 10
would place the damage differently, and they do not.

**Therefore H1 is falsified for this case: chunked streaming decode reproduces
whole-sequence decode.** Consistent with this, both the whole path and the streaming path
use the same attention window (72) and the streaming path trims its K/V cache coherently.

Scope of the claim, deliberately narrow: CLI versus CLI, one process, one sentence, one
voice, ARM, no lane, no `QWEN_DECODER_BATCH`, no per-slot state. It exonerates the streaming
decode MECHANISM; it does not exonerate the server's decoder lane, the batched decoder, or
per-slot decoder state. And if the codec tokens themselves differ between CLI and server, the
decoder is irrelevant regardless.

## 4. Ranked hypotheses — updated with the 3.4 measurement

| # | hypothesis | category | first evidence that would confirm | first evidence that would kill it |
|---|---|---|---|---|
| ~~H1~~ | ~~Streaming decoder chunk continuation corrupts audio at chunk boundaries~~ | 4 | — | **FALSIFIED locally, see 3.4**: chunked reproduces whole at mel-corr 0.99975 and is chunk-size invariant. Only the server-specific decoder paths (lane, batched decoder, per-slot state) remain untested. |
| H2 | Server stopping policy (3.1) truncates or over-extends vs CLI | 1/3 | frame counts differ for the same text/seed; divergence at/near the end | frame counts identical and divergence is mid-utterance |
| H3 | Per-slot state installed at admission differs from CLI pre-generation state | 1 | a KV/dec_x/position/tcl field differs right after ADMIT_INSTALL | all installation-time state hashes equal |
| H4 | Batched Talker/CP kernels differ numerically even at B_eff=1 | 2/3 | state equal at install, first Talker step differs, divergence index early | first Talker step bit-equal |
| H5 | Prefix-cache reuse injects a prompt prefix built under different conditioning | 1 | failure disappears with `QWEN_PREFIX_CACHE=0`; first request behaves differently from later ones | identical failure with the prefix cache disabled |

With H1 falsified, **H2 and H3 move to the top**. H2 because the stopping policy is
demonstrably a different function in the two paths (section 3.1) and its input is a token
length, which makes it look language-dependent without being so. H3 because installation is
the remaining place where the server builds state the CLI never builds.

The externally reported observation that CLI and server already differ at concurrency 1 is
consistent with this: it removes true batching from the primary suspects and, combined with
3.4, points upstream of the waveform decoder. That observation is not this author's
measurement and is recorded as reported, not as established.

## 5. Instrumentation reality check (important, and easy to trip over)

**`QWEN_DUMP_CODES` does not cover the batched server.** It is written in `qwen_cp_predict`
(the scalar/CLI Code Predictor). The batched server uses `qwen_batch_cp_predict`, which has
no equivalent dump. Any "compare codec tokens CLI vs server" step therefore needs a per-slot
dump added to the batched CP path first. `QWEN_DUMP_CODE0` has the same limitation (it lives
in the CLI generate loop).

`tests/decoder_standalone` (`test_decoder_standalone.c`) replays a `QWEN_DUMP_CODES` file
through the decoder, which makes hypothesis H1 testable WITHOUT the server at all.

## 6. The next single discriminating test

Section 3.4 already ran the offline decoder A/B that this section originally proposed, and
it came back negative. The next single test therefore belongs to the execution track and is
the codec-token bifurcation:

> For the same model, voice, language, text, seed and sampling configuration, at B_eff = 1:
> are the generated codec token IDs identical between CLI and server?

If they DIFFER: the waveform decoder is out of scope entirely (3.4 plus this). Report the
first differing frame index; early points at installation/conditioning (H3, H5), late points
at the stopping policy (H2).

If they are IDENTICAL: the defect is in the server-specific decoder paths that 3.4 did NOT
cover — the decoder lane, `QWEN_DECODER_BATCH`, per-slot streaming state — and the next test
is to disable them one at a time.

Blocker to be aware of before instrumenting: `QWEN_DUMP_CODES` lives in `qwen_cp_predict`
and does not cover `qwen_batch_cp_predict`, which is what the batched server runs (section 5).

## 7. Classification table for the returning evidence

| bucket | fact | rules out | makes more likely | next single test |
|---|---|---|---|---|
| A | divergence BEFORE the first Talker step | kernel and decoder causes | prompt/prefill/installation/metadata (H3, H5) | field-by-field diff of installed slot state vs CLI pre-generation state |
| B | state exact, Talker diverges at an early step | state installation | execution/kernel/sampling state (H4) | same request with the batched kernels forced to the scalar path |
| C | Talker equivalent, CP diverges | Talker and prefill | CP init/execution | CP substep comparison at the first divergent frame |
| D | codec tokens identical, WAV differs | everything upstream | decoder/stream continuation (H1) | section 6, offline |
| E | B_eff=1 clean, B_eff>=2 fails | single-slot causes | true batching / ragged / shared state | B_eff=2 with the second slot idle vs active |
| F | CustomVoice fails, Base OSS clean | generic server-state causes | conditioning path specific | compare conditioning state only, before touching the scheduler |
| G | both model paths fail at the same boundary | model-path-specific causes | generic server continuation/state | continue at that boundary, not elsewhere |

## 8. Stop rules

Stop and report at the FIRST proven divergence boundary. Do not keep descending.
If KV differs right after slot installation, do not look at the decoder. If codec tokens
differ, do not touch the waveform decoder. If codec tokens match exactly, do not spend time
on Talker prefill. If CustomVoice fails and Base OSS does not, compare conditioning state
before any scheduler or kernel work. No "while I'm here" fixes.

## 9. Evidence table template

| run | model path | voice | lang | text id | seed | precision | server cfg | B_eff | frames | first differing codec frame | WAV differs | ASR transcript | human PASS/FAIL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

ASR is never the sole oracle: it has its own error floor and it is language-dependent. Every
row needs the human field filled before it is used as a verdict.

## 10. Regression bank (to build when the execution track has a boundary)

Small and OSS-safe: 8-12 short Italian sentences covering weekday/date words, doubled
consonants, accented final vowels, common elisions, longer multisyllabic words and
punctuation boundaries, plus 4-6 English controls of comparable token length. Fixed seed,
fixed voice, both model paths. Store text, seed, model path, voice path, CLI and server
results, codec divergence index, ASR transcript and a human PASS/FAIL field.

## 11. Relation to C12-WIN

This defect blocks a final PRODUCT QUALITY claim for any new runtime path. It does NOT block
the spec 11A/12 kernel microbenches (decoder timing and numerical parity, independent of
language quality) nor a diagnostic server A/B. No new runtime path is promoted until the
defect is understood well enough to show the treatment does not worsen it.
