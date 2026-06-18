# Expressivity: emotion & prosody control

> ## ★ The strongest lever: the native-instruct recipe (1.7B, ear-validated 2026-06-08)
> `--instruct` was thought to "barely change delivery" — that was a usage artifact. It works **well**
> with two conditions the obvious approach misses:
>
> 1. **Temperature 1.1–1.3** (not the 0.5 default, and NEVER `-T 0`). Greedy/low temp picks the flattest
>    prosody → emotion suppressed. At T1.1–1.3 the model expresses it and even adds spontaneous
>    paralinguistics (a sad lead-in *"hmm… let me tell you…"*). T1.1/1.3 stay clean — you can push them.
> 2. **Write the instruct in ENGLISH (or Chinese) — the model's training languages — NOT the spoken
>    language.** Qwen3-TTS's instruct-following is EN/ZH-centric; an Italian instruct is barely understood
>    (stays flat), while an **English instruct on Italian speech** emotes strongly and keeps the speech
>    cleanly Italian. (Chinese instruct ≈ a hair stronger, esp. sad.)
>
> ```bash
> ./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.2 \
>   --instruct "Speak deeply sad and heartbroken, a slow broken voice on the verge of tears." \
>   --text "Allora ti racconto una cosa successa oggi." -o sad.wav
> ```
> The `ryan` preset cross-speaks EN/IT/FR/ES well.

> ## ★★ Emotion on a CLONED voice
>
> > **DEFAULT (updated 2026-06-18): x-vector-only from a tiny 8 KB `.bin`.** The ICL graft documented below
> > works, but the ICL `ref_codes` carry the reference RECORDING'S room acoustics (a faint "muffled metallic /
> > reverb") that gets re-injected every generation — and an `.expr` amplifies it. The speaker x-vector carries
> > identity WITHOUT the room → clean clone, identity preserved, and more force headroom for the `.expr`
> > (`-T 1.3`, weight ~1.6–2.0). Make the `.bin`: `python3 tests/qvoice_to_xvec.py voices/x.qvoice`, then
> > `--load-voice voices/x.bin --xvector-only`. Keep `--icl-only` (below) only for maximum timbre mimicry from
> > a studio-clean reference. Full analysis: `docs/csp-ft-emotion.md`, `docs/icl-graft-portability.md`.
>
> ### The `--icl-only` graft (ear-validated 2026-06-09) — alternative, max mimicry
> A normal `.qvoice` **resists instruct**: loading it SWAPS the CustomVoice (CV) weights for the
> base-cloned weights, and the base weights follow an instruct ~**3.8× weaker** (measured: emotion
> relative-shift 25% preset → 6.6% qvoice). The voice is faithful but emotionally frozen — and forcing it
> (activation `--ml-steer`, or scaling the WDELTA via `--voice-strength`) either collapses energy or, on a
> clone, distorts timbre/language ("Romanian-accent" / gender flips), because emotion & identity share weights.
>
> **The fix: keep the CV weights, graft only the clone's ICL prefix.** `--icl-only` loads a `.qvoice`'s
> speaker-embedding + reference codec tokens (the in-context "voice prompt") but **SKIPS the WDELTA
> weight-swap**. The instruct-capable CV weights stay intact, so the model does intelligent *mix-and-match*:
> timbre/identity from the ICL prefix × emotion from the instruct.
>
> ```bash
> # Galatea (a cloned .qvoice) speaking Italian, genuinely SAD — stays Galatea, adds a natural sigh:
> ./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_17b.qvoice --icl-only \
>   -l Italian --seed 42 -T 0.9 \
>   --instruct "Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears." \
>   --text "Non posso credere a quello che mi è successo oggi, è incredibile." -o sad.wav
> ```
> Measured vs the frozen `.qvoice`: instruct response 6.6%→**12.6%** (≈2×), per-emotion movement reaches
> preset grade (angry mel-vs-neutral 0.57→**0.35**), and emotions are **distinct** again (qvoice ang≈sad
> 0.74 collapsed → graft 0.42; the matrix even tracks real relations, happy≈excited). At T0.9 it produced a
> spontaneous trembling sigh before "è incredibile" — preset-grade paralinguistics on a clone.
> **⚠ The sigh is seed-specific: this exact text + instruct + `-T 0.9` + `--seed 42` = the "ehhhh… è
> incredibile" sospirato (ear-re-confirmed 2026-06-10, user: "top!"). Change the text/instruct/seed and
> the spontaneous sigh may not appear — keep all four fixed to reproduce it.**
>
> **Recipe knobs (ear-validated on Galatea):**
> - `--icl-only` (load the `.qvoice` on the **CustomVoice** model, NOT base) + **English** instruct.
> - **Temp ≈ 0.9 is the sweet spot.** T1.5 over-creates: stretches/drags words and adds spurious "eh…"
>   interjections. T0.9 = clean, faithful, controlled natural expression. (It speaks the text faithfully —
>   it does NOT paraphrase.)
> - **Sad: just use a plain concise sad instruct at T0.9.** The low temp already avoids the word-stretching;
>   adding an explicit "keep a natural pace, don't drag" to the instruct counter-intuitively made it SLOWER
>   (ear-tested) — keep the instruct short.
> - **Limit:** a naturally calm cloned voice gives *irritated/passive-aggressive* anger and a *veiled*
>   sadness, not screaming rage (instruct response is ~half a preset's) — a structural ceiling of cloning.
>
> **What the graft CAN vs CANNOT mod (Galatea, ear-validated):** the ICL prefix is a strong speaker-IDENTITY
> anchor, so the graft changes things *orthogonal* to identity but not identity itself:
> - ✅ **Language** — Galatea wishing happy birthday in Spanish / German, keeping her timbre (cross-lingual
>   clone, the official "Clone" feature).
> - ✅ **Emotion / tone** — sad, angry (passive-aggressive), happy, excited — distinct and preset-grade.
> - ❌ **Speaker identity** — gender / age / dialect barely move (the ICL prefix pins the speaker); higher temp
>   doesn't unlock it, just adds artifacts. "young" → mostly faster speech; "feminine" on a male clone → an
>   effeminate version of the same man, not a new female. (NB the `it_galatea_fasol` reference is actually
>   **Riccardo Fasol, a male narrator** — "Galatea" is the book title — so our clone is correctly a man.)
>
> **Older clone levers (kept, default-off, for experiments):** CP control-vectors, `--voice-strength`
> (targeted WDELTA scaling via `--vs-layers`), multi-layer `--ml-steer`. The `--icl-only` graft above is the
> recommended way to emote a clone. NEXT (PLAN.md "official models re-analysis"): graft directly on a fresh
> `--ref-audio` clone, on the VoiceDesign model, and whether CV/VD can ingest ref-audio natively.

Qwen3-TTS's built-in `--instruct` at the default temp barely changes delivery (see the recipe above to
fix that). This engine also adds two CPU-side levers that act directly on the
**Code Predictor** (the stage that carries texture/prosody), giving controllable,
audible delivery. Both default off → the normal path is bit-identical (no overhead).
NOTE: these control-vector emotions are MILDER than the native-instruct recipe above and work
cross-model (incl. 0.6B, no instruct); prefer the recipe on 1.7B presets.

> **Practical, ear-validated per-mood & per-language recipes** (joy=excited, sad=slow+pauses,
> annoyed=angry+roughness, cross-language notes, dead-ends): see
> [expressivity-recipes.md](expressivity-recipes.md).
>
> **Writing long-form (audiobooks/podcasts)?** Use **inline markup** — one text with
> `[sad]`/`[excited]` mid-text emotion switches, `[pause:400ms]` pauses and `[sigh]`/`[huff]`
> fillers, rendered in one pass: see [markup.md](markup.md).

## `--emotion <name>` — delivery presets

A calibrated palette of emotion/tone "control vectors" ships in `presets/emotions/`.
Each preset already has its recommended strength baked in, so the name alone sounds right:

```bash
./qwen_tts -d qwen3-tts-0.6b --text "Here is the news everyone was waiting for." --emotion news -o out.wav
```

| preset | delivery / use case |
|---|---|
| `happy` | cheerful, upbeat — light podcast, good news |
| `excited` | high-energy, hype — sports, announcements |
| `eager` | keen, "let me tell you the news!" |
| `proud` | dignified, proud newsreader |
| `sad` | sorrowful, downcast |
| `gloomy` | dark, somber — true-crime, grim news |
| `news` | clear, authoritative anchor — political talk |
| `dramatic` | suspenseful, storytelling — audiobooks |
| `calm` | soft, soothing — late-night radio, meditation |

Tune the strength globally with `--steer-weight` (multiplies the baked calibration):

```bash
./qwen_tts ... --emotion happy                 # calibrated (recommended)
./qwen_tts ... --emotion happy --steer-weight 1.3   # push harder
./qwen_tts ... --emotion sad   --steer-weight 0.6   # softer
```

**Blend** moods by listing several `name[:scale]` separated by commas:

```bash
./qwen_tts ... --emotion "happy:0.5,proud:0.5"   # warm + dignified
```

Presets are **cross-model** (the Code Predictor is identical on 0.6B and 1.7B), so a
vector captured on 1.7B works unchanged on 0.6B. An Italian palette lives in
`presets/emotions/it/` (point at it with `QWEN_EMOTION_DIR=presets/emotions/it`).

> The strength is also a **mood crossfade**, not just intensity — pushing a direction far
> can land on a neighbouring emotion. The shipped weights are the sweet spots we tuned by ear.

## Compound moods — one name sets every knob

A single mood name can drive the *whole* recipe — vector **and** steer-weight **and** roughness
**and** volume **and** rate — because some emotions are prosody, not just a steering direction
(sadness is slower + quieter; joy is brighter + faster + louder). These live in a manifest
(`qwen_tts_emotion.c`):

```bash
./qwen_tts ... -l Italian --emotion joy       # excited @2.6 + rate 1.10 + volume 1.10
./qwen_tts ... -l Italian --emotion sad       # sad @2.0 + rate 0.84 + volume 0.90
./qwen_tts ... -l Italian --emotion annoyed   # angry @2.6 + roughness 0.32 + brisk
```

| mood | recipe (vec · weight · roughness · vol · rate) |
|---|---|
| `joy` | excited · 2.6 · — · 1.10 · 1.10 |
| `excited` / `proud` / `eager` / `dramatic` | self · 2.0–2.2 |
| `news` | proud · 2.0 |
| `calm` | calm · 1.6 · — · 0.95 · 0.96 |
| `sad` | sad · 2.0 · — · 0.90 · 0.84 |
| `gloomy` | gloomy · 2.0 · — · 0.90 · 0.85 |
| `annoyed` | angry · 2.6 · 0.32 · 1.05 · 1.05 |
| `stern` | angry · 2.6 · 0.28 · 1.05 · — |
| `angry` | angry · 2.6 · 0.40 · 1.05 · 1.05 |

Any explicitly-passed flag **overrides** the baked value, e.g. `--emotion joy --steer-weight 1.4`.
The resolver is **language-aware**: with `-l Italian` it pulls from the centered palette
(`it_centered/` then `it/`) automatically. If a mood's vector is missing for the active language
(e.g. `angry` only exists in the IT-centered palette), steering is skipped but the prosody knobs
(roughness/volume/rate) still apply. A blend/scale spec (`happy:0.5,proud:0.5`) bypasses the
manifest and steers the raw presets.

## Prosody knobs — `--volume` and `--rate`

Independent of `--emotion`, two post-synthesis knobs:

- `--volume <f>` — output gain (1.0 = unchanged; 1.1 louder, 0.9 softer). Pure PCM gain; also
  applied per-chunk in `--stream`.
- `--rate <f>` — **pitch-preserving** speaking rate via in-engine WSOLA (>1 faster, <1 slower).
  No ffmpeg dependency. Not applied in `--stream` mode (the time-stretch needs the full buffer).

```bash
./qwen_tts ... --rate 0.9 --volume 0.95   # slower & softer, pitch unchanged
```

### Voice-specific tuning

Steering directions are captured on the *preset* voice (ryan) distribution. Most tones
transfer cleanly onto other preset speakers and custom `.qvoice` voices, but two caveats:

- **A direction can over-steer on a different voice** → lower it with `--steer-weight`
  (e.g. a soft-spoken cloned voice may need `--emotion excited --steer-weight 0.5`).
- **Some tones depend on a vocal register the voice doesn't have.** A bright `happy`
  needs an upbeat register; a soft, low-energy voice (e.g. a calm narrator clone) simply
  can't reach it — even a *natively captured* happy fades. Use `eager`/`excited` for
  upbeat delivery on such voices instead. For a voice that genuinely warrants its own
  calibration, capture a native palette and drop it in `presets/emotions/<voice>/`
  (point `--emotion` at it with `QWEN_EMOTION_DIR=presets/emotions/<voice>`).

## `--roughness <0..1>` — texture / grit

An orthogonal knob: blends a 2-bit copy of the FFN `down` output into the
high-precision one (`down` is the causal driver of the "rough/aggressive" texture).
Dials in grit/anger/worn-voice continuously, and **combines with any `--emotion`**:

```bash
./qwen_tts ... --roughness 0.3     # light edge
./qwen_tts ... --roughness 0.6     # strong, aggressive
./qwen_tts ... --emotion gloomy --roughness 0.4   # grim + gritty
```

Works under bf16/int8/int4 (the q2 copy is built lazily from the bf16 weights).

## Building your own presets

A preset vector is `mean(cp_x | instruct) − mean(cp_x | neutral)`, captured at the
single Talker→Code-Predictor injection point. Capture is env-gated (`--instruct` needs
the 1.7B model):

```bash
# 1. capture an instruct run and a neutral run (same text + seed)
QWEN_STEER_CAPTURE=/tmp/a.vec ./qwen_tts -d qwen3-tts-1.7b --text "$TXT" \
    -I "Speak with intense anger" --seed 42 -s ryan -l English -o /dev/null
QWEN_STEER_CAPTURE=/tmp/b.vec ./qwen_tts -d qwen3-tts-1.7b --text "$TXT" \
    --seed 42 -s ryan -l English -o /dev/null

# 2. build the direction (optionally bake a weight with --scale)
python3 tests/steer_make.py /tmp/a.vec /tmp/b.vec presets/emotions/angry.vec --scale 0.7

# 3. use it
./qwen_tts ... --emotion angry          # resolves presets/emotions/angry.vec
./qwen_tts ... --steer-vector /tmp/custom.vec --steer-weight 0.7   # any path
```

Use a **multi-sentence** capture text so content averages out and only the delivery
remains. `tests/steer_palette.sh [model] [lang] [outdir]` rebuilds the whole calibrated
palette in one shot.

`.vec` format: `'QSTV'` magic (uint32 LE) + int32 dim + dim×float32 (dim = CP hidden = 1024).
