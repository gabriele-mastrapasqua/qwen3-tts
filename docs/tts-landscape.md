# Open TTS landscape — how others do emotion, paralinguistics, cloning

> Intelligence-gathering, NOT a plan to replace Qwen3-TTS. Goal: learn HOW comparable open/open-weight
> TTS models implement the things our expressivity engine struggles with — **paralinguistic laugh/sigh
> discrimination, cross-lingual nonverbal transfer, emotion-intensity dosing, and making cloned voices
> emote** — and mine techniques that fit our small-LoRA-on-top approach. Survey: deep-research, 2026-06-14.

## TL;DR — the three things to steal

1. **Paralinguistic discrimination is a DATA problem, not a steering problem.** Every model that does
   reliable laugh-vs-sigh uses **distinct, trained, inline text tags** on text-aligned tagged data. This
   confirms our own next-retrain recipe (clean, distinct-per-marker, aligned events). We are not using the
   wrong mechanism — we just need cleaner per-marker data.
2. **EmoSteer-TTS** — training-free **activation steering** with a single continuous `alpha` (sign+magnitude)
   — is a clean, dosable, no-retrain analog to our control-vectors that directly targets our energy-collapse
   and intensity-dosing problems. Cheap to prototype (an activation add at inference).
3. **IndexTTS2** — **emotion↔timbre disentanglement** (gradient-reversal + a separate style prompt) — is the
   architectural answer to our **clone-resistance** (cloned `.qvoice` voices that refuse to emote).

## Per-model comparison (the axes we care about)

| Model | Arch | Emotion control | Paralinguistics (laugh/sigh) | Cloning | Cross-lingual emotion | License |
|---|---|---|---|---|---|---|
| **Orpheus** | AR LLM + SNAC codec | implicit / prompt | ✅ **distinct inline tags** `<laugh> <sigh> <chuckle> <gasp> <yawn> <cough> <sniffle> <groan>` | zero-shot from N text-speech pairs; reliability ↑ with #pairs (accepts clone-resistance) | not shown | Apache-2.0 |
| **CosyVoice2** | AR LLM + FSQ codec + flow | NL instruct + `[emotion]` tags | ✅ **burst** `[laughter]`/`[breath]` mid-text **AND sustained** `<laughter>XXX</laughter>` ("speak WITH laughter") + `<strong>` | zero-shot ref | ❌ emotion eval **Chinese-only** | Apache-2.0 |
| **Fish-Audio S2** | AR LLM | prompt | ✅ **15,000+** bracket tags (`[laughing] [sigh] [inhale] [whisper] [chuckle]`) | zero-shot ref | n/a | partly closed |
| **Bark** | AR LLM | implicit | ✅ `[laughs] [sighs] [gasps] [clears throat]` — but **probabilistic placement** | preset/ref | n/a | MIT |
| **IndexTTS2** | AR + dur model | **3 modes**: text "soft-instruction" (**Qwen3-1.7B LoRA**), emotion-ref audio, emotion vector | partial | zero-shot + **emotion/timbre DISENTANGLED (GRL)** → clones CAN emote | partial | Apache-2.0 |
| **EmoVoice** | LLM (Qwen2.5-0.5B) | **freestyle NL prompt** straight into LLM (`Say this with emotion of [desc]`) | — | ref | — | open + EmoVoice-DB (40h/7 emo) |
| **EmoSteer-TTS** | steering on DiT TTS | **training-free activation steering, 1 continuous `alpha`** (α=0 none, α>0 toward, α<0 away/erase) | — | — | — | research |
| **ProEmo** | prosody-mod | **GPT-4 predicts pitch/energy/dur scaling** (global tone + per-word) + learned intensity ranker | — | — | — | research |
| **dots.tts** (rednote) | **fully continuous AR**, AudioVAE 48k + Qwen2.5-1.5B + flow-matching DiT + CAM++ x-vec | ⚠️ **implicit only** (inferred from text; no lever) | ❌ not supported | **continuation** (ref + transcript = best SIM) + x-vec | not shown | **Apache-2.0** |
| XTTS / F5-TTS / StyleTTS2 / Parler / Kokoro / MeloTTS | various (AR/flow/diffusion) | mostly ref-audio or none (Parler = NL caption) | mostly ✗ | ref / preset | — | mostly permissive |

## Where our engine (Qwen3-TTS) stands

- **Emotion**: we use the **converging field idiom** — natural-language instruct fed into the LLM (EN-instruct
  + temperature) **plus** per-language **LoRA** packs. IndexTTS2's "soft-instruction = Qwen3-1.7B LoRA" is
  essentially our recipe → we're on the right track, not behind.
- **Paralinguistics**: we do inline tags + a small LoRA (the field-standard mechanism). Our gap (laugh/sigh
  discrimination) is a **data-cleanliness** gap, not a mechanism gap.
- **Cross-lingual nonverbals**: **UNSOLVED across the whole open field** — nobody benchmarks laugh-vs-sigh
  cross-lingual. Our event-splice augmentation (manufacture data by splicing real events into per-language
  carriers) is genuinely **frontier**; there is nothing to copy, only to push.
- **Cloning**: dots.tts + IndexTTS2 both confirm **continuation cloning (ref audio + TRANSCRIPT)** beats
  x-vector-only — validating our `--ref-text` full-ICL path (surface + recommend it).
- **Fidelity bar**: **dots.tts beats Qwen3-TTS** on Seed-TTS-Eval (WER 2.95/SIM 79.2 vs 3.07/74.5) and
  MiniMax multilingual (SIM 83.9, leads 19/24 langs) — but with **no explicit emotion/paralinguistic control**.
  So our differentiator is **controllable expressivity**, theirs is raw fidelity/latency.

## Ideas worth stealing — prioritised

1. **[near-term, fits our recipe] Clean, distinct-per-marker tagged data** for laugh/sigh discrimination.
   Adopt CosyVoice2's **burst vs sustained** two-form idea (`[laughter]` burst vs `<laughter>…</laughter>`
   speak-with-laughter). Source: NonverbalTTS/NVTTS (text-aligned, 10 NV types). → our paralinguistic retrain.
2. **[cheap prototype] EmoSteer-style activation steering** — one continuous `alpha`, training-free, applied
   as a forward-hook add on a residual stream; α<0 even ERASES emotion. Targets our dosing + energy-collapse.
   Open Q: transfers from their DiT decoder to our AR-LM + 12Hz-codec? Worth an experiment.
3. **[bigger, post-release] IndexTTS2 emotion/timbre disentanglement (GRL)** for clone-emotion. Likely needs
   the speaker rep trained disentangled (not retrofittable onto a baked WDELTA `.qvoice`) → a clone-format
   change. The principled fix for clone-resistance.
4. **[noted] ProEmo's LLM-predicted pitch/energy/duration scaling** (global + per-word) — an alternative
   intensity-dosing lever if instruct+temp+LoRA isn't enough.

## Sources
Orpheus (github.com/canopyai/Orpheus-TTS) · CosyVoice2 (arxiv 2412.10117) · Bark (github.com/suno-ai/bark) ·
Fish-Audio S2 (fish.audio/s2) · IndexTTS2 (arxiv 2506.21619, index-tts.github.io) · EmoVoice (arxiv 2504.12867) ·
EmoSteer-TTS (arxiv 2508.03543) · ProEmo (arxiv 2501.06276) · NonverbalTTS/NVTTS (arxiv 2507.13155,
hf.co/datasets/deepvk/NonverbalTTS) · dots.tts (github.com/rednote-hilab/dots.tts, Apache-2.0).
