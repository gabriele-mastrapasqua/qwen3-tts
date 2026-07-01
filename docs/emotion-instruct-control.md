# Instruct control — strength & prosody (the `--instruct` lever, 1.7B)

What the free-text `--instruct` can and cannot do **on top of** THE emotion recipe, and how to write it well.
Ear-validated 2026-06-30 (`samples/tests/2026-06-30_structured-instruct/` + `..._instruct-strength/`).

> **Scope.** `--instruct` is **1.7B only** (the 0.6B ignores it). It matters only in the **COMBINE / instruct
> path** — cloned voices, or a preset used *with* an instruct. **Preset pure-STEER emotion (THE recipe) needs no
> instruct and is unchanged by any of this.** The real emotion dial is still steer weight + expr; the instruct is a
> secondary flavour that rides on top.

## TL;DR
1. **Write the instruct in plain English, vivid and strong** — not as a parameter schema. Qwen reads it as *mood*,
   not as fields.
2. **Strength is the knob that works.** `strong` is the reliable level; `very-strong` often pushes further (anger
   gets raspier/angrier). Overdoing wording can destabilise (noise) on some emotions — trust the ear.
3. **Speed is promptable** in plain English (`"speak faster/slower"`) — a mild but real lever. Pitch-up a little.
4. **Do NOT use a slot template** (`VoiceStyle:/Tempo:+15%/Pitch:higher`). Qwen does not parse it; `Tempo:+40%`
   even comes out *slower*. It's a dead end — see below.

## What works

### Instruct strength — the primary instruct lever ✅✅
A stronger, more vivid free-form instruct pushes emotion harder (up to a point). Keep a per-emotion library at two
levels and pick by ear:

| emotion | `strong` (reliable default) | `very-strong` (extra push — anger especially) |
|---------|-----------------------------|-----------------------------------------------|
| sad | Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears. | Speak utterly devastated and broken, voice trembling and cracking, sobbing, barely able to get the words out through the tears. |
| joy | Speak with bright, radiant joy, light and warm, smiling through every word. | Speak overflowing with ecstatic, explosive joy, laughing with delight, breathless with excitement, bursting with happiness. |
| anger | Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage. | Speak in an absolutely furious, explosive, screaming rage, voice cracking with violent anger, completely losing control. |
| fear | Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread. | Speak in sheer terror, voice shaking violently, gasping and panicking, barely able to breathe with fear. |
| disgust | Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you. | Speak with overwhelming, sickened revulsion, recoiling and almost gagging, utter contempt dripping from every word. |
| surprise | Speak with sudden astonishment and surprise, gasping and caught off guard. | Speak in total shock, gasping loudly, completely stunned and overwhelmed, unable to believe what just happened. |

The `strong` column is exactly what `tests/emo_suite.sh` already ships (`INS`). The `very-strong` column is the new
keeper. Movement metrics are non-monotone and **undercount raspy/intense voice** → the ear overrides the number.

### Speed via plain English ✅ (mild)
Instruct-only sweep on ryan/EN: `"speak much more slowly"` → 4.80 s, anchor 4.40 s, `"much faster"` → 3.60 s,
`"extremely fast"` → 3.52 s — monotone, ~27 % span. `"speak in a higher/brighter voice"` raises F0 ~+18 %.
Compose these hints into the instruct when you want pacing/pitch, e.g.
`--instruct "Speak with bright joy, and speak a little faster."`

## What does NOT work

- **Slot / parameter template** — ❌. `"VoiceStyle: happy. Tone: cheerful. Pitch: higher. Tempo: +15%. Intensity:
  medium-high. Expression: smiling."` The model does not parse the slots as parameters:
  - `Tempo:+40%` → *slowest* output (inverted); `-20%` → fastest. Not a speed dial.
  - `Pitch:lower` ≈ `Pitch:higher` (both in the F0 noise floor).
  - `Intensity` / `Expression` → no measurable effect.
  - Emotionally it's a wash vs a plain vivid instruct (no upgrade). The structure adds nothing; write prose instead.
- **Loudness** — output is loudness-normalised; `"speak loudly"` doesn't raise RMS (though `"shouting"` changes
  pitch/effort/timbre).

## How to write a good instruct (rules of thumb)
- Plain vivid English prose, present-tense imperative: *"Speak … , voice … , as if …"*.
- One clear emotion, concrete physical cues (voice low/heavy, trembling, cracking, breathless) beat adjectives.
- Add a pacing phrase in words if you want it (`"… and speak a little faster"`), never `Tempo:+X%`.
- Escalate to the `very-strong` wording for more push; back off if it gets noisy/unstable on that emotion.

## Reproduce
- `tests/structured_instruct_test.sh` — the full §8.8 A/B (slot template vs free-form) + per-slot + NL-prosody.
- `tests/measure_prosody.py <wav…>` — objective dur / median-F0 / RMS-dB (+ `--move ref out` for mel-movement).
- Instruct-strength sweep + the reusable library: `samples/tests/2026-06-30_instruct-strength/README.md`.

See also: **`docs/emotion-THE-recipe.md`** (the emotion recipe this rides on top of).
