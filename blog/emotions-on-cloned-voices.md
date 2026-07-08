---
title: "Emotions on a cloned voice: a 25 MB graft, a steering vector, and a lot of dead ends"
published: false
description: "How we got sad / happy / angry / fearful speech to work on cloned voices in a pure-C Qwen3-TTS engine — a 25 MB 'graft' clone that keeps the emotion levers alive, a steering + fine-tune recipe hard-won after many dead ends, and — because emotion is a vector — blended 'dyad' emotions plus switching emotion mid-sentence from a single prompt."
tags: machinelearning, tts, c, audio
---

*Part of [qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) — a pure C inference engine for Qwen3-TTS.*

## TL;DR

Qwen3-TTS ships 9 neutral preset speakers. No emotion control, and cloning a voice used to mean a huge file that couldn't emote at all. We changed both:

- **Small clones.** A cloned voice is now a **~25 MB `.qvoice` "graft"** — small enough to share, and, crucially, built so the emotion machinery still works on it.
- **Emotions on *any* voice.** One flag — `--emotion <sad|joy|anger|fear|disgust|surprise>` — works on **presets and cloned voices**, in every supported language.
- **Mixing emotions.** Because emotion is a *direction* you can add, summing two gives a new one: seven **dyads** (`contempt`, `awe`, `nostalgia`, …) fall out for free. And you can **switch emotion mid-sentence from a single prompt** with inline `[emotion]` tags.

Getting there took an embarrassing number of dead ends. This post is the honest map of what failed and what finally worked.

---

## Why emotion on a clone is hard

The neutral clone was the easy part: feed 30 seconds of audio, an ECAPA-TDNN encoder extracts a speaker embedding, and the model reproduces the timbre. But that clone is **frozen and flat.** It says the words in the right voice with no feeling.

Two forces fight you when you try to add emotion to a clone:

1. **Emotion and timbre live in the same weights.** Push the model toward "angry" and the timbre drifts — you get an angry *stranger*, not an angry *you*.
2. **A cheap clone throws away the levers.** The smallest clone formats (a 4 KB x-vector, a KV-cache prefix) bolt a voice onto the model but lose the internal state the emotion tools need to hook into.

So the clone format and the emotion method are not independent problems. They have to be designed together.

---

## The 25 MB graft: small *and* emotable

We landed on a **graft** `.qvoice` (`--icl-only`): instead of shipping the whole retrained model, it keeps the CustomVoice transformer weights and stores just the delta needed to *be* your voice. About **25 MB**. That's the sweet spot:

- Small enough to attach to an email or check into a repo.
- Preserves full prosody (it's not a lossy 4 KB summary).
- **Keeps the emotion levers alive** — because the CustomVoice weights are still present, the steering and fine-tune hooks have something to grab.

(For reference: a bit-identical clone is ~785 MB, a shareable one ~16 MB, a postcard x-vector ~4 KB — but only the graft keeps the instruct/emotion controls working. Trade-offs, all measured.)

---

## The graveyard of methods that didn't work

Before the recipe that ships, we tried — and abandoned — a lot. Writing them down so nobody (including future us) re-derives them:

- **τ-vectors / task-arithmetic (`.vec`).** Compute an "emotion direction" by float-space arithmetic between neutral and emotional fine-tunes, add it at inference. Elegant on paper; muddy and timbre-shifting in the ear.
- **x-vector emotion injection.** Bolt an emotion onto the 4 KB speaker vector. Too little state to carry it.
- **Per-language dense fine-tunes.** Full FT of layers 16–26 per language. Big, brittle, and it averaged emotions together instead of letting you *select* one.
- **Seed palettes.** Curate "good seeds" per emotion. Fragile and non-portable across voices.
- **Graft-emotion, per-language EXPR-COMBINE variants.** Each solved one case and broke another.

Every one of these produced audio. None of them produced *reliable, selectable, timbre-preserving* emotion on a cloned voice. They're archived — the useful thing they left behind is the recipe below.

---

## What actually works: steer for presets, COMBINE for clones

The shipped system is two hooks used together:

1. **Activation steering.** A tiny, speaker-and-language-agnostic direction added to the residual stream at layers 21–25 at inference time. It nudges "emotion" without touching timbre. The vectors are a few KB and committed to the repo.
2. **CSP fine-tune (`.expr`).** A small weight-delta band (a few layers, LoRA-style) trained on real emotional speech — the **Emozionalmente** Italian emotion corpus (CC-BY 4.0; Catania, Wilke & Garzotto, PoliMi, IEEE TASLP 2025). Romance languages transfer from the Italian pack; other languages get their own small pack.

The one rule:

- **Preset voice → pure steering.** Clean in every language, nothing else needed.
- **Cloned voice → COMBINE** — the language `.expr` fine-tune **plus** the steering vector **plus** an English instruct prompt, applied *together*. Neither alone is enough on a clone; together they push emotion hard enough to be heard while the fine-tune keeps it in-distribution so the timbre survives.

That "together" is the whole trick. The steering vector supplies a clean emotional *direction*; the fine-tune supplies the emotional *texture* the clone lacks; the instruct prompt sets the *strength*. Remove any one and it degrades.

The result is one flag:

```bash
# preset voice
./qwen_tts --emotion sad -s ryan -l English --text "I can't believe he's gone."

# your own 25 MB cloned voice — same flag
./qwen_tts --qvoice me.qvoice --emotion joy -l Italian --text "Ce l'abbiamo fatta!"
```

Native preset per language under the hood (Japanese, Korean, Chinese, Romance, Russian…), so the emotion lands naturally instead of fighting the language.

---

## Mixing emotions: dyads, and switching mid-sentence

Here's the fun part. If emotion is a *direction* in activation space, then directions **add**. Take the "anger" vector and the "disgust" vector, sum them 50/50, and you get something that is neither — a coherent, recognizable **contempt**. It just works, and seven blends (Plutchik's "dyads") fall out of the six primaries we already had. The full menu:

| `--emotion` | Kind | What it is |
|---|---|---|
| `sad` · `joy` · `anger` · `fear` · `disgust` · `surprise` | primary | the six base emotions (synonyms like `happy` / `angry` work too) |
| `contempt` | dyad | anger + disgust → sneering disdain |
| `awe` | dyad | fear + surprise → hushed wonder |
| `nostalgia` | dyad | joy + sad → bittersweet fondness |
| `disapproval` | dyad | surprise + sad → let-down reproach |
| `remorse` | dyad | sad + disgust → guilty regret |
| `outrage` | dyad | anger + surprise → indignant shock |
| `despair` | dyad | fear + sad → hopeless dread |

No new training, no new capture — just `dyad_mix.py a.qlsteer:0.5 b.qlsteer:0.5`, and seven new `--emotion` values appear. (One thing the ear caught: `joy`-paired blends over-drive on long English sentences, so `nostalgia` ships 40/60 sad-leaning. Vectors add — but the mix ratio still matters.)

Then the demo that makes people lean in: **many emotions from one prompt.** Write `[emotion]` tags inline and the engine switches sentence by sentence, in a single generation, clean at the seams:

```bash
./qwen_tts -s ryan -l English --text \
  "[contempt] Oh, sure, that's a brilliant idea. [nostalgia] We used to spend every summer by the sea. [despair] And now there's nothing left."
```

One file, three emotions, no splicing — the steering vector is simply swapped per sentence while the voice stays the voice. `[neutral]` resets.

**One system, everywhere — which meant deleting our own earlier work.** We'd built an older, weaker per-sentence emotion path on a different (CP-level `.vec`) mechanism. It's retired. Now the CLI `--emotion` flag, the inline `[emotion]` tags, *and* the HTTP server's `emotion` field all route through the **same** steering recipe — so a dyad you find on the command line behaves identically in a server request, and a REST client can stream `[joy]…[sad]…` markup and get per-sentence emotion for free. One recipe, three surfaces.

---

## Paralinguistics: inline events — 🧪 work in progress

Emotion is prosody; **paralinguistics** — a laugh, a sigh, a yawn — is an *event*. We ship a handful as inline tags. Each fires **in one generation, in the voice's own timbre** — no splice (a spliced laugh sounds like a *different person* laughing; the tag becomes a validated onomatopoeia *inside* the sentence instead, so it's your clone doing it):

| Tag | Event |
|---|---|
| `[laugh]` | a real chuckle |
| `[sigh]` | a sigh |
| `[yawn]` | a yawn |
| `[wow]` | a "wow!" interjection |
| `[giggle]` | a sly giggle (best on its own — pairing it with `joy` over-drives it) |
| `[scoff]` | a dismissive *tsk* |

```
./qwen_tts --text "That's hilarious [laugh] I can't even. [sigh] Okay, back to work."
```

**Fair warning — treat this as alpha.** It's hit-or-miss across voices and languages (laughs and sighs land best), and it's parked for now rather than under active development. But it works often enough to be fun, and it's very much worth a try: if it breaks in an interesting way on your voice or language, that's exactly the kind of bug report that makes it better. Finding a universal onomatopoeia-per-event was its own long hunt, and there's surely more to find.

---

## Takeaways

- **Clone format and emotion method are one problem.** The 25 MB graft exists specifically so the steering/fine-tune hooks survive.
- **One clean method beats five clever ones.** τ-vectors, x-vector emotion, dense per-language FT, seed palettes — all archived in favor of *steer + fine-tune, together*.
- **On a clone, layer the levers.** Steering direction + fine-tune texture + instruct strength. Individually weak, together strong.
- **Keep events in-timbre.** A spliced laugh is a stranger; an inline one is you.
- **Emotion is a vector — so it composes.** Directions add (dyads) and can be swapped per sentence (inline switching). Wire every surface — CLI, inline, server — to the *one* recipe and the feature multiplies for free.

It's all pure C, CPU by default, and the emotional-expressivity `.expr` packs are fetched on demand from HuggingFace. Clone your voice once, then make it *feel* something — after, admittedly, a lot of experiments that didn't.
