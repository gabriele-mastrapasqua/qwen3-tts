# Emotion instruct-prompt library

A reusable bank of **English instruct prompts** for expressive TTS, organized by family.
Taxonomy inspired by the Dawizzer ComfyUI emotion node (75 presets); the *prompts* are ours
because on this engine the working emotion lever is **`--instruct` in English**, not sampling
knobs (those only add arousal — see `docs/emotion-THE-recipe.md`).

## The recipe (what actually works)

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian \
  --instruct "<vivid English instruction from below>" \
  -T 1.1 --seed 42 --text "<your text>" -o out.wav
```

- **Preset voice** (`-s ryan`) follows instruct strongest. **Cloned voice — DEFAULT**: load an
  8 KB **x-vector `.bin`** with `--xvector-only` (`--load-voice voices/X.bin --xvector-only`) at
  **T1.3** — it keeps identity without the reference recording's room reverb, so it's cleaner and
  has more force headroom. Make the `.bin` with `python3 tests/qvoice_to_xvec.py voices/X.qvoice`.
  *Alternative*: load a `.qvoice` with `--icl-only` (the ICL graft) for max timbre mimicry from a
  clean ref — emotion works but is capped at *moderate* (no screaming/sobbing). See
  [`docs/csp-ft-emotion.md`](csp-ft-emotion.md) and [`docs/icl-graft-portability.md`](icl-graft-portability.md).
- **English instruct even for non-English speech** — the model's instruct-following is EN/ZH-centric.
- **Temperature 1.1–1.3** (never `-T 0` = flat). **Seed-audition**: re-run with several `--seed`
  values and keep the best take — spontaneous paralinguistics (sighs, "ehhh…") emerge on *some* seeds.
- Keep the spoken **text emotionally neutral** when you want the *instruct* (not the words) to carry
  the emotion, so it reads cleanly.

## Anger

| emotion | English instruct |
|---|---|
| annoyed | Speak mildly annoyed and irritated, a short-tempered edge under the words. |
| angry | Speak angrily, sharp and confrontational, clearly upset. |
| furious | Speak with furious, seething anger, voice hard and forceful, barely holding back. |
| enraged | Speak in a towering rage, explosive and venomous, spitting the words out. |
| bitter | Speak with bitter, resentful sarcasm, cold and wounded. |
| vengeful | Speak with cold, vengeful menace, promising payback, slow and deliberate. |
| frustrated | Speak with mounting frustration, exasperated, on the edge of giving up. |

## Sadness

| emotion | English instruct |
|---|---|
| disappointed | Speak quietly disappointed, let-down and deflated. |
| sad | Speak with a sad, sorrowful, downcast tone, voice low and heavy. |
| melancholic | Speak with deep melancholy, wistful and aching, lingering on the words. |
| hopeless | Speak in flat, hopeless despair, drained of all energy. |
| defeated | Speak utterly defeated, beaten down, voice small and resigned. |
| guilty | Speak with quiet guilt and regret, unable to meet the listener's eyes. |
| remorseful | Speak with heavy remorse, deeply sorry, voice thick with regret. |
| ashamed | Speak with shame, halting and embarrassed, wishing to disappear. |

## Joy

| emotion | English instruct |
|---|---|
| content | Speak warmly content and at ease, a soft satisfied smile in the voice. |
| happy | Speak happily, bright and warm, smiling through the words. |
| excited | Speak with bright, bubbling excitement, fast and energetic. |
| ecstatic | Speak in pure ecstatic joy, overflowing, almost breathless with delight. |
| enthusiastic | Speak with eager enthusiasm, lively and infectious. |
| relieved | Speak with deep relief, a long exhale, tension melting away. |
| proud | Speak with warm pride, chin up, savoring the accomplishment. |
| playful | Speak playfully and lightly, a teasing grin in the voice. |

## Fear / Stress

| emotion | English instruct |
|---|---|
| nervous | Speak nervously, a little shaky and uncertain, words tripping slightly. |
| anxious | Speak with anxious worry, tight and restless, fearing the worst. |
| fearful | Speak with fear, hushed and wary, glancing over the shoulder. |
| terrified | Speak terrified, trembling, voice breaking with dread. |
| panicked | Speak in rising panic, fast and breathless, barely keeping control. |
| frantic | Speak frantically, words tumbling out, desperate and scattered. |
| overwhelmed | Speak overwhelmed, struggling to keep up, voice strained thin. |
| desperate | Speak with raw desperation, pleading and urgent, clutching at hope. |

## Calm / Low energy

| emotion | English instruct |
|---|---|
| calm | Speak in a calm, even, reassuring tone, unhurried. |
| relaxed | Speak relaxed and easygoing, loose and untroubled. |
| serene | Speak with serene, peaceful stillness, soft and grounded. |
| bored | Speak flatly bored, disinterested, dragging the words. |
| tired | Speak tired and weary, low energy, words slowing down. |
| exhausted | Speak utterly exhausted, drained, each word an effort. |

## Social / Attitude

| emotion | English instruct |
|---|---|
| sarcastic | Speak dripping with sarcasm, mock-sweet, clearly meaning the opposite. |
| mocking | Speak mockingly, taunting and derisive, making fun of the listener. |
| condescending | Speak condescendingly, patronizing, talking down as if to a child. |
| dismissive | Speak dismissively, can't-be-bothered, waving the matter away. |
| contemptuous | Speak with cold contempt, looking down with disdain. |
| sympathetic | Speak with gentle sympathy, soft and caring, offering comfort. |
| apologetic | Speak apologetically, contrite and sincere, making amends. |
| pleading | Speak pleadingly, imploring, begging the listener to understand. |
| confident | Speak with calm confidence, assured and steady, no doubt at all. |
| arrogant | Speak arrogantly, self-important and superior, certain of being right. |
| smug | Speak smugly, self-satisfied, quietly gloating. |
| insecure | Speak insecurely, hesitant and self-doubting, seeking reassurance. |

## Mood / Character

| emotion | English instruct |
|---|---|
| teasing | Speak teasingly, light and mischievous, poking fun affectionately. |
| flirtatious | Speak flirtatiously, warm and playful, a charming lilt. |
| seductive | Speak seductively, low and slow, intimate and inviting. |
| mysterious | Speak mysteriously, hushed and intriguing, hinting at secrets. |
| ominous | Speak ominously, dark and foreboding, a warning beneath the words. |
| menacing | Speak menacingly, quiet and dangerous, an unspoken threat. |
| suspicious | Speak suspiciously, guarded and probing, not quite trusting. |
| paranoid | Speak with paranoid unease, hushed and twitchy, certain someone is listening. |

## Physical states

| emotion | English instruct |
|---|---|
| breathless | Speak breathlessly, as if having just run, catching the breath between words. |
| whispering | Speak in a soft whisper, intimate and quiet, barely above a breath. |
| shouting | Speak loudly and forcefully, projecting as if across a room. |
| drunk | Speak tipsy and loose, words slightly slurred and meandering, overly familiar. |
| in_pain | Speak through pain, tight and wincing, breath catching. |

> **Compose / blend tip:** for nuance, layer two compatible directions in one instruct
> (e.g. "*bitter and exhausted*", "*nervous but trying to sound confident*"). Unlike the
> sampling-knob averaging (which cancels), an instruct can hold a genuine semantic blend.
