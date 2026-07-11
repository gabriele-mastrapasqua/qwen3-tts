# Inline expressive markup (audiobooks & podcasts)

Write one text with **inline tags** and the engine renders an expressive, multi-emotion
take in a single pass — different moods per sentence, paragraph-level pauses, and small
paralinguistic fillers (sighs, huffs). The tag style follows the modern AI-TTS convention
(ElevenLabs / Bark): **English tags in square brackets**, placed inline, switchable mid-text.

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
  --text "I couldn't believe it. [joy] We actually won! [pause:500ms] [sad] But it meant leaving everyone behind. [sigh]" \
  -o out.wav
```

You don't need a special flag: if `--text` contains recognized tags it is rendered as
expressive markup automatically. (`--compose "..."` does the same explicitly.)

> **Mid-text `[emotion]` tags need the 1.7B model.** On 0.6B they are silently ignored (emotion is a
> 1.7B feature). Pauses and paralinguistic fillers below work on both models.

## Tags

| tag | effect |
|---|---|
| **primaries** `[sad] [joy] [anger] [fear] [disgust] [surprise]` (synonyms: `[happy] [angry] [rage] [afraid] [disgusted] [surprised] [sadness]`) | switch the emotion for the text that follows (steering vector @ the ear-validated weight; 1.7B only) |
| **dyads** `[contempt] [awe] [nostalgia] [disapproval] [remorse] [outrage] [despair]` | blended Plutchik emotions (two steering directions summed; 1.7B only) |
| `[neutral]` (`[none]`, `[normal]`) | back to plain, unmodified delivery |
| `[sigh]` `[sighs]` | sigh of relief (`Hah…`) |
| `[ahh]` `[relief]` | pleasure/relief exhale (`Haaa…`) |
| `[phew]` | big tired relief (`Uao…`) |
| `[hmm]` | pensive "hmm" (`Hmmm…`) |
| `[mmm]` | soft assent "mmm" (uses CN 嗯 — cleaner/less smug) |
| `[hmpf]` | closed "mmmm" |
| `[mah]` | dismissive "mah" (very Italian) |
| `[uhm]` | tired/bored drawl |
| `[laugh]` `[laughs]` | a real chuckle (`Eheh…`) — best in Italian |
| `[haha]` | short laugh (`Haha!`) — best in English |
| `[heh]` | smug "eh eh" |
| `[ouch]` / `[ahi]` | sharp pain — English / Italian |
| `[huff]` `[ugh]` | irritated huff |
| `[pause:400ms]` `[pause:1s]` `[pause:0.5]` `[break:300ms]` `[0.5]` | insert a pause (ms or seconds; bare number = seconds) |

> Many fillers are **language-dependent** (the same string produces a different sound per
> language): e.g. `[laugh]`/`Eheh` laughs in Italian, `[haha]`/`Haha` in English. Discover
> and extend the set with `tests/sound_suite.sh` (mass-generate → listen → bake winners).
>
> **Chinese phonetic characters are a clean source** of paralinguistic sounds even under a
> non-Chinese language: 嗯 → "mmm" (used by `[mmm]`), 哈哈/嘿嘿/呵呵 → laughs, 唉 → a weary
> sigh. Some emoji also leak a sound (😂 → a faint sigh). Probe more via the suite.

> Paralinguistic tags are **soft, un-steered** onomatopoeia tuned by ear (a leading `h` adds
> breathy aspiration). They're approximations, not recorded breaths — and a few are
> language-dependent (e.g. `[laugh]` lands as a real laugh in English, a sigh elsewhere).

- Tags are **case-insensitive** and **always English** (`[sigh]`, not `[sospiro]`).
- An emotion tag stays active until the next emotion tag or `[neutral]`.
- Text before the first tag is spoken neutrally.
- **Unrecognized** `[...]` is left as literal text — a stray bracket won't break your script.
- The emotion recipe is in
  [emotion-THE-recipe.md](emotion-THE-recipe.md).

## How it renders

Each span (a run of text under one emotion) is synthesized **separately** with its own
recipe, then all spans are concatenated into one WAV with the pauses you asked for. Because
every span is **model-generated** (same voice, same 24 kHz codec), the joins are seamless —
this is *not* audio splicing from a reference, so there are no phase/timbre artifacts.

Adjacent spoken spans get a small default gap (`--compose-pause`, default `0.12s`) so words
don't collide; explicit `[pause:…]` tags add exactly the silence you specify.

## Paralinguistic fillers — what they are (and aren't)

`[sigh]`/`[hmm]`/`[laugh]`/etc. are **approximations**, not recorded breaths. The trick: a short
aspirated onomatopoeia (`"Hahh…"`, `"Hmmm…"`, `"Hehhh…"` — a leading `h` adds breathiness) is
synthesized with **no emotion steering**, a gentle slowdown and low volume. (Steering a short,
time-stretched vowel goes metallic/"growl", so fillers are pure soft prosody.) They read
convincingly as tiredness/relief/amusement in context, but they are synthesized vowels, not
true non-verbal breaths (the model has no real `<breath>` token — see the dead-ends in
[emotion-THE-recipe.md](emotion-THE-recipe.md)).

## Tuning

- Per-span moods use their **baked recipe** weight/rate/volume (calibrated by ear). To bias a
  whole render louder/slower, add global `--volume`/`--rate` (they post-process the final mix).
- For Italian, the language-aware resolver automatically uses the centered palette — just pass
  `-l Italian`.
- Want a custom default gap between spans? `--compose-pause 0.25`.

## Examples

```bash
# Audiobook beat: setup (neutral) -> reveal (joy) -> turn (sad + sigh)
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
  --text "The letter sat on the table for days. [pause:400ms] [joy] When she finally opened it, she gasped. [pause:600ms] [sad] It was the goodbye she'd feared. [sigh]" \
  -o scene.wav

# Italian, explicit --compose form (| is an optional hard span break)
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian \
  --compose "[anger] Te l'avevo detto! [pause:300ms] | [neutral] Va bene, ricominciamo. [sigh]" \
  -o dialogo.wav

# Tired character: huff, beat, resigned line
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
  --text "[huff] [pause:300ms] [sad] Fine. I'll do it myself." -o tired.wav
```
