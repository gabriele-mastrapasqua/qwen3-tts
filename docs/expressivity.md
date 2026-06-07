# Expressivity: emotion & prosody control

Qwen3-TTS's built-in `--instruct` ("speak angrily") barely changes delivery — a known
limitation of the model. This engine adds two CPU-side levers that act directly on the
**Code Predictor** (the stage that carries texture/prosody), giving controllable,
audible delivery. Both default off → the normal path is bit-identical (no overhead).

> **Practical, ear-validated per-mood & per-language recipes** (joy=excited, sad=slow+pauses,
> annoyed=angry+roughness, cross-language notes, dead-ends): see
> [expressivity-recipes.md](expressivity-recipes.md).

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
