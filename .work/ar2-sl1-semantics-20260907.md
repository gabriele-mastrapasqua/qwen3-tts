# AR-2 / SL-1 semantic verification

Date: 2026-09-07
Implementation reference: `8d9ff3de9bb75009d42d43c3875925c16e9e2540`
Scope: read-only comparison of the official known-text streaming layout with the C
prompt and step path. No runtime change is included in this note.

Official sources checked:

- [Qwen3-TTS inference API](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
- [Qwen3-TTS model construction and Talker forward](https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/refs/heads/main/qwen_tts/core/models/modeling_qwen3_tts.py)

## Verified official behavior

`non_streaming_mode=False` is a known-text prompt schedule, not live network
streaming. The official wrapper still tokenizes the complete request before calling
the model. `generate_voice_clone` defaults to `False`; `generate_custom_voice` and
`generate_voice_design` default to `True`. The underlying conditional-generation
`generate()` also defaults to `False`.

For a non-ICL request the official sequence is:

1. optional instruct text projection;
2. the three role tokens;
3. codec control prefix: `codec_nothink` + think BOS + think EOS for `Auto`, or
   `codec_think` + think BOS + language + think EOS for an explicit language;
4. optional speaker embedding, then codec PAD/control prefix and codec BOS;
5. first assistant text token combined with codec BOS as the last prefill position;
6. with streaming layout, the remaining text projections followed by `tts_eos` are
   supplied one per generated Talker step, then `tts_pad` forever;
7. each generated step combines the generated codec-group embedding with that
   trailing text hidden. The codec EOS remains the generated stop token.

For non-streaming mode, all assistant text plus `tts_eos` are prefilled with codec
PAD and the final `tts_pad + codec_bos` position is retained. This is the current C
layout.

For ICL, the official text track is `reference text + requested text + tts_eos` and
the codec track is `codec_bos + reference codec frames`. Streaming mode overlays
the two tracks for the common prefix; if text is longer, the unpaired text tail is
returned as trailing text hidden. If codec is longer, text is padded with `tts_pad`.
Non-streaming mode instead places the complete text track before the codec track.

The Talker uses ordinary causal positions/RoPE over the resulting sequence. There is
no separate RoPE reset for the trailing text. `tts_eos` is a hidden text-track input;
codec EOS is still the generated termination token. The official Talker adds the
trailing text hidden when `generation_step` is within its length and adds `tts_pad`
after exhaustion. The wrapper stops at the first generated codec EOS and trims ICL
reference audio outside the Talker generation result.

## Current C behavior at the frozen HEAD

| official streaming operation | current C behavior | required C change |
|---|---|---|
| Non-ICL prefill ends after role/control prefix + first text token + codec BOS | `qwen_tts_generate()` materializes and prefills all text tokens + `tts_eos` with codec PAD, then final `tts_pad + codec_bos` (`qwen_tts.c:1320-1336`) | Add an opt-in known-text layout that pre-fills only the first text token/codec-BOS position and stores the remaining text hidden sequence. |
| Remaining non-ICL text is consumed one hidden per Talker step | CLI and server always add `tts_pad` to generated codec embeddings (`qwen_tts.c:1679-1689`, `RECORD_FRAME_AND_EMBED`) | Add per-context/per-slot trailing text position and add trailing hidden, then `tts_pad` after exhaustion. |
| ICL streaming overlays reference codec frames with reference+requested text, with overflow as trailing hidden | C currently emits complete reference/request text track, then codec BOS/reference frames (`qwen_tts.c:1283-1318`): the official non-streaming arrangement | Build the aligned common prefix and retain the unpaired text tail; preserve reference codec frames and output trimming semantics. |
| ICL/non-ICL share the official prefix/control construction | C has the same broad role/control ordering, but uses locally resolved `language_id`, speaker/x-vector rules, `graft_mode`, `QWEN_SPK_SCALE`, and local tokenizer/template handling | Compare control-token IDs and speaker-injection conditions per mode; do not change those policies as part of the first layout patch. |
| `tts_eos` is followed by `tts_pad` once text is exhausted; codec EOS still stops generation | C has local EOS suppression/ramp/top-k policy and currently no trailing EOS hidden; it stops on codec EOS | Preserve local codec-EOS policy; add exactly one text `tts_eos` in the trailing sequence and pad thereafter. Quality/audio gates are required because schedule changes model behavior. |
| Positions/RoPE follow the shortened prefill then one step per codec frame | C uses `ctx->kv_len` as the next absolute position and therefore will follow the new sequence automatically only if the shortened prefill and trailing step sequence are implemented consistently | Keep one monotonic KV position stream; no RoPE reset or special text position. Add position/token structural tests. |
| Known text is prepared before generation; this is not live incremental text | C has no trailing-text state in `qwen_tts_ctx_t` or batch slots and server admission calls the full `qwen_tts_generate()` | Add request-owned state, deep-copy it into server slots/helper results, and keep network input unchanged. Live text remains research-only. |

## Prefix-cache and resumability correction

The official layout does not make the local prefix cache a resumable prefill cursor.
At present C builds the complete `input_embeds` array first, then uses either a
matching prefix cache or a complete prefill. `qwen_talker_prefill()` returns only
after all new positions/layers are complete; no layer/token cursor survives. The
first SL-1 implementation must therefore either disable full-prompt delta reuse for
the experimental layout or define a cache key/entry that is valid for the shortened
prefill. It must not interpret `prev_prefill_len` as a pause/resume mechanism.

## Scope and gates

SL-1 is initially **known text only**, default-off and experimental. It changes the
model-visible schedule and is not expected to be bit-identical to the current
non-streaming control. Validation must include structural token/position parity
against the official construction, deterministic local replay, audio quality and
long/short prefill scaling. Server support is required before any serving claim, but
live network text, append-after-generation, and resumable fixed-prompt prefill are
out of scope for this slice.
