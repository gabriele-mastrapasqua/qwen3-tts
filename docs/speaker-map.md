# Speaker map of Qwen3-TTS — where timbre, language/prosody, and emotion live

A readable map of **where each "sector" of a voice lives** in the Qwen3-TTS pipeline, from timbre →
language/prosody → emotion → final audio, and **how the preset voices (e.g. `ryan`) work**. Grounded in the
engine code + the measured activation maps (`docs/prosody-map.md`, captured with `QWEN_ACT_MAP`).

Model = 1.7B CustomVoice (CV): a **28-layer causal Talker** (Qwen3 LLM, residual stream, hidden 2048) →
**Code Predictor** (MTP, 15 passes/frame) → **Speech Decoder** (convolutional codec renderer, 480× upsample).

## The map

| Stage (in flow order) | Sector it controls | What it does | Where (measured) | How to change it |
|---|---|---|---|---|
| **Input: speaker x-vector** | **Timbre / identity** (seed) | ECAPA-TDNN embedding (dim 2048) injected as one *continuous codec-slot token* before the Talker | pre-Talker input | **DEFAULT clone path (2026-06-18):** `--load-voice voices/X.bin --xvector-only` (an **8 KB** file) |
| **Talker L00–L06** (early) | **Language + pacing/rhythm** | intonation, timing, "melody" — **strongest for Italian** | L00–06 (IT pacing L03/L06/L07/L08 ≈ 43–46 %) | a **DENSE** fine-tune that includes the early band |
| **Talker L07–L12** (mid) | **prosody (peak) + IT prosody-identity + the language difference** | where prosody is realized; EN-vs-IT separation concentrates **L07–10** | L07–12 + final | dense fine-tune |
| **Talker L16–L26** | **EMOTION** | instruct/expressivity competence; emotion magnitude (mid L06–11) + **emotion identity L21–25** | L16–26 | the emotion `.expr` / dense FT |
| **Talker final hidden** | prosody + emotion magnitude | aggregates, read out into the Code Predictor | final | — |
| **Code Predictor** (15 passes/frame) | fine acoustic detail | refines the 16 codec tokens per frame | downstream | (the `.expr` does not touch it) |
| **Speech Decoder** (ConvNet, 480×) | **renders codec tokens → waveform** | turns codec tokens into audio | downstream | (the `.expr` does not touch it) |

**Cross-cutting facts:**
- **Timbre/identity** is seeded by the **x-vector at the input** and carried by the codec tokens the Talker
  emits. The `.expr` / FT on the Talker layers **never touches timbre**, so you can change emotion/language
  without losing the voice.
- **The x-vector alone is the DEFAULT clone path (2026-06-18):** `--load-voice X.bin --xvector-only`. It
  carries identity **without** the reference recording's room acoustics — unlike ICL `ref_codes`, which
  re-inject a faint "muffled metallic / reverb" every generation that an `.expr` then amplifies. The 8 KB
  x-vector in a lite voice is **byte-identical** to the one in the full 2.8 GB qvoice (cosine 1.0000), so the
  clean-vs-metallic difference is the ref_codes, not the embedding. x-vector-only is clean and **tolerates a
  higher `.expr` weight**; keep ICL (`--icl-only`) for **max timbre mimicry** from a studio-clean ref. See
  `docs/icl-graft-portability.md` and `docs/csp-ft-emotion.md`.
- **Paralinguistics** (*where* to place a sigh/laugh) is **structural and lives EARLY (L00–12)**, outside the
  emotion band — which is why an L16–26-only paralinguistic LoRA couldn't *place* nonverbals.

## Timbre deep-dive — does the x-vector touch the decoder? No.

The Speech Decoder consumes **only the codec tokens** (`qwen_speech_decoder_decode(ctx, codes, ...)`) — it
**never sees the speaker embedding**. It is a **speaker-agnostic, fixed renderer**: it faithfully turns *any*
codec-token sequence into audio. So:

- Timbre is **seeded** by the x-vector (input) → it **conditions the Talker** to emit speaker-appropriate
  codec tokens → the decoder **renders** them. The "timbre at the end" is just the decoder realizing tokens
  that **already encode** the speaker — not the decoder storing per-speaker weights.
- **We do not (and need not) touch the decoder** to change a voice. By design (codec-LM TTS), speaker identity
  lives in the **codec-token sequence**, not in decoder weights; the codec decoder is universal/frozen. This is
  why a tiny x-vector is enough to swap a voice, and why a heavy 3 GB WDELTA dump is unnecessary for identity.

## How the preset voices work (e.g. `ryan`)

A preset speaker is a baked **speaker ID (0–8)** → a fixed speaker embedding inside the CV weights. The CV
Talker was trained mostly on **ZH / JP / EN** voices, so:
- **`ryan` (EN)** does Italian/French/Spanish well via a cross-lingual **"switch"**; **`vivian` (ZH)** does
  Italian less natively (Chinese-accented Italian).
- Preset **emotion = the instruct alone** (a ChatML user turn) conditioning **L16–26**, plus temperature — **no
  weight change**. (See `docs/expressivity.md`.)

## Productization: separate LANGUAGE (heavy, shared) from SPEAKER (tiny x-vector)

The scalable way to add voices is **not** a per-voice 3 GB WDELTA. It is two composable artifacts:

| Artifact | What it is | Size | Scope |
|---|---|---|---|
| `*.cvft` / dense `.expr` | a **DENSE per-language fine-tune** of the CV (melody/timing/emotion/paralinguistics), trained **voice-agnostic** (many speakers) | ~186 MB dense (or a 4 GB checkpoint) | **one per language**, shared |
| `*.xvec` | a per-speaker **x-vector** (ECAPA identity) | **KB** | **one per voice** |

**Compose at inference (DEFAULT, 2026-06-18):** `base CV + --expr <language dense FT> + --load-voice <speaker
x-vector .bin> --xvector-only + instruct + temp` — at **T1.3, weight ~1.6–2.0** (`w2.5/T1.3` svaria; joy/disgust
stay flat with weight → training ceiling, retrain `top_k=4`). Make the speaker `.bin` from any qvoice or ref:
```bash
python3 tests/qvoice_to_xvec.py voices/X.qvoice
# or, clone straight to a .bin from a ref recording (the engine, not the helper):
./qwen_tts -d qwen3-tts-1.7b-base --ref-audio ref_24k_mono.wav --xvector-only --save-voice voices/X.bin
```

Keep the language FT **voice-agnostic** (train on many speakers / preset IDs) so *any* x-vector composes on top;
do **not** bake one clone's timbre into it. Fidelity from x-vector-only cloning is ~80–90 % — the trade buys a
natively-Italian, expressive voice. Going from "moves a bit" → "everything WOW" is a **data + capacity** problem:
a richer, varied Italian dataset (prosody + paralinguistics) + a **DENSE** fine-tune on a **wider** band (L00–12
language/prosody **+** L16–26 emotion) — DENSE, *not* the low-rank broad-band that flattened clones.

> **🔴 Gotcha (see `regression_lora.md` §7):** a DENSE `.expr` is a bf16 **bit-delta** → valid **only on
> CV-intact weights** (preset / `--icl-only` graft / small-ICL). Applied on a **full-WDELTA** load it corrupts
> the bits → metallic garble. The graft (x-vector, no ref_codes anchor) is the recipe; full-WDELTA is not.

See also: `docs/prosody-map.md` (the per-layer measurements), `regression_lora.md` (dense-vs-LoRA + the recipe),
`docs/expressivity.md` (the instruct recipe), `docs/icl-graft-portability.md` (x-vector-only default vs ICL),
`docs/csp-ft-emotion.md`.
