# Emotion Vector — the working recipe (disentangled, float task-arithmetic)

> **STATUS 2026-06-16: PARTIAL — promising, NOT done.** The disentangled emotion-vector method is the right
> direction (it's what real systems use: arXiv 2507.03382 / IndexTTS2 — a speaker/language-agnostic emotion
> vector on a FROZEN backbone, NOT a full weight FT). Ear-test (ryan, vivian, Italian, α=1.0):
> - ✅ **anger, sad** = clean + expressive on ryan.
> - ❌ **fear (and other high-arousal emotions)** glitch on ryan at α=1.0 → "ba bu ba bu" + glitches.
> - 🟡 **vivian** holds + emotes but is **accented** (vivian is a Chinese preset → "Chinese speaking Italian"),
>   no word glitches.
> **CONFIRMED CAUSE (fear diagnostic):** FEAR no-expr (instruct+temp only) = clean, even adds a spontaneous
> scared "ehhh". The τ at EVERY α (0.3/0.5/1.0) GLITCHES — "gomani" for "domani" (pronunciation!), metallic
> inter-word glitch, eats words. So the narrow **L16-26 τ DAMAGES prosody/pronunciation** for prosody-heavy
> emotions — it was trained ONLY on the emotion band, MISSING the PROSODY band L00-12 (see
> [`prosody-map.md`]). This is "yesterday's error" (emotion-only band).
> **Two consequences:** (1) on PRESETS, `instruct+temp` ALONE is the answer — the τ is unnecessary and HURTS.
> (2) the τ's real use is the CLONE (which resists instruct). To make τ usable it must be **WIDE-band (L0-27)**
> so it carries prosody → retrain emotion-FT + neutral-FT at `--layers 0-27`, re-extract τ_wide, test ON THE CLONE.
> The disentanglement (emotion−neutral) still cancels the pronunciation drift at wide band → prosody+emotion,
> no corruption (the hypothesis to verify).

## What it is (one paragraph)

An **emotion task vector** `τ = θ_emotion_FT − θ_neutral_FT`, where both models are dense-FTs of the SAME
multi-speaker data (one on emotional speech, one on neutral). The subtraction **cancels everything they
share** — speaker mix, recording domain, and crucially the **pronunciation drift** toward the dominant
training language — leaving **~pure emotion**. Applied on the clean base, `θ = base + α·τ`, it makes the
voice emote WITHOUT rewriting its pronunciation. Emotion is **language-agnostic**; pronunciation stays the
base's. The specific emotion is chosen by the **instruct**, not by the file.

## How many files + which params (the practical answer)

- **ONE `.expr` file** covers **ALL emotions** for a language family. The file is the emotion *capability*
  vector; the **instruct** picks which emotion (happy/anger/sad/fear/disgust/surprise). You do NOT need one
  file per emotion.
- To **use** it at inference you need exactly:
  - `--expr presets/expr/italian_emovec_float.expr` (the τ)
  - `--expr-weight 1.0` (intensity — see note: **1.0 is the sweet spot**, 0.5 can half-correct → slight drift)
  - `--instruct "<vivid ENGLISH instruction>"` (the emotion selector; English/ZH, model is EN/ZH-centric)
  - `-T 1.1` (temperature; ≤1.3, 1.5 slurs), `-l Italian`, `-s <preset>`
- To **build** the τ you need (transiently) **3 model checkpoints**: base CV, emotion-FT, neutral-FT
  → produces the 1 `.expr`. After that only the `.expr` matters.

## Use it — commands (α = 1.0)

```bash
TAU=presets/expr/italian_emovec_float.expr
COMMON="-d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 --expr $TAU --expr-weight 1.0"

# neutral (no instruct)
./qwen_tts $COMMON --text "Domani vado al mercato a comprare il pane e la frutta." -o neu.wav
# anger
./qwen_tts $COMMON --instruct "Speak with hot, furious anger, sharp and forceful." \
  --text "Domani vado al mercato a comprare il pane e la frutta." -o anger.wav
# happy / sad / fear / disgust / surprise — same, swap the instruct:
#   "Speak happily, bright and warm, smiling through the words."
#   "Speak with a sad, sorrowful, downcast tone, voice low and heavy."
#   "Speak with fear, tense and trembling, your voice wary."
#   "Speak with physical disgust, repulsed and recoiling."
#   "Speak with surprise, startled and taken aback, held through the whole sentence."
```

`--expr-weight` note: **use 1.0.** Counter-intuitively 0.5 can sound slightly worse (a brief "foreign"
attack) because τ also carries the pronunciation correction; at half strength it's only half-applied. Full
strength = full correction = clean. Push >1.0 only to over-dramatize (may distort).

## Build it — commands

```bash
# 1) two dense FTs on the SAME multi-speaker set (frozen except L16-26 + text_projection):
#    θ_emotion : all rows, emotional instructs   (training/expressivity-lora/dgx_sft_expr_lang.py)
#    θ_neutral : neutral rows only, empty instruct
# 2) pull both checkpoints + the base locally, then FLOAT task-arithmetic -> .expr:
python3 tests/tau_arith.py \
    qwen3-tts-1.7b \                       # base CV (frozen backbone)
    qwen3-tts-1.7b-expr-multitag \         # θ_emotion checkpoint dir (model.safetensors)
    qwen3-tts-1.7b-expr-neutral \          # θ_neutral checkpoint dir
    presets/expr/italian_emovec_float.expr --alpha 1.0 --lang Italian
```

## ⚠️ The bug we fixed (do NOT repeat)

`tests/expr_extract.py emotion neutral` (bit-pattern subtraction of two INDEPENDENT fine-tunes) → **pure
noise**. expr_extract subtracts bf16 BIT PATTERNS, valid only for near-identical models (emotion vs the base
it came from). Two separate FTs land in different exponent octaves → garbage. **`tests/tau_arith.py` does
the subtraction in FLOAT32** then re-encodes the (new−base) delta → valid. (The millions of int16 "overflow"
warnings are a red herring — the mod-2¹⁶ wrap is lossless; ignore the count, judge by ear.)

## Why this beats the dense full-FT (the journey)

- Full dense FT on mixed multilingual emotion data **corrupted Italian pronunciation** ("foreigner speaking
  Italian") because it was 97.7% English and rewrote the SHARED pronunciation weights. Adding a language tag
  helped gentle emotions (sad) but not aggressive ones (anger) — data-starved (~84 IT anger clips vs the
  literature's 1–30 h/emotion). See [`../training/expressivity-lora/MULTISPEAKER_PIPELINE.md`].
- The emotion *vector* avoids this: backbone frozen, pronunciation untouched, low-data, language-safe.

## Open / next

- **FUTURE TODO — cloned small voices.** Does `base + τ` hold on a **cloned `.qvoice` (small-ICL / graft)**?
  The clone's x-vector is out-of-distribution; emotion has resisted there. The disentangled τ (pronunciation-
  safe) is the best shot yet — must be tested. This is the real prize (clone emotion).
- Per-language: τ is currently from the EN-heavy multi-speaker set (language-agnostic by construction). If a
  language needs more, build a per-language τ (e.g. Italian-only EMOVO emotion − EMOVO neutral).
- Tune the canonical α per emotion if needed (default 1.0).

## Links
- Deep history + reproduce: [`../training/expressivity-lora/MULTISPEAKER_PIPELINE.md`]
- Roadmap / decisions: `PLAN.md` (local-only) — "SESSION 2026-06-16" block.
- Builder: [`../tests/tau_arith.py`]; FT: [`../training/expressivity-lora/dgx_sft_expr_lang.py`].
