# Custom Voices (`.qvoice`)

Save a cloned voice to a `.qvoice` file once, then reuse it forever — on the Base model,
on the CustomVoice model (with `--instruct` style control), on the server, with streaming.
No re-extraction needed.

## Delta Format (Recommended)

The Delta format is the recommended way to create custom voices. It produces output that is
**bit-identical** to a direct Base model clone, while running on the faster CustomVoice model.

### How It Works

Base and CustomVoice models share **99.98% identical transformer weights**. The only meaningful
differences are in the codec embedding table and a few special tokens. The Delta format exploits
this by storing only the per-weight differences (int16 deltas, LZ4 compressed). At load time,
these deltas patch the CustomVoice weights to exactly match the Base model — producing
PCM-level bit-identical output.

For the full technical analysis, see [blog/cross-model-voice-analysis.md](../blog/cross-model-voice-analysis.md).

### Creating a Delta `.qvoice`

Requires both Base and CustomVoice models downloaded. The key flag is **`--target-cv`**,
which points to the CustomVoice model directory.

**Always specify `-l` with the language of the reference audio** — it's saved in the
`.qvoice` and auto-applied when loading.

```bash
# One-time setup: download both models
./download_model.sh --model base-small    # 0.6B-Base (for voice extraction)
./download_model.sh --model small         # 0.6B-CustomVoice (target for deltas)

# Create the delta .qvoice
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio mario.wav -l Italian \
    --voice-name "Mario" --target-cv qwen3-tts-0.6b \
    --save-voice mario.qvoice
```

For the 1.7B model:

```bash
./download_model.sh --model base-large
./download_model.sh --model large

./qwen_tts -d qwen3-tts-1.7b-base --ref-audio mario.wav -l Italian \
    --voice-name "Mario" --target-cv qwen3-tts-1.7b \
    --save-voice mario_17b.qvoice
```

### Using a Delta `.qvoice`

Only the CustomVoice model + the `.qvoice` file are needed at runtime. The Base model
is only required during creation.

```bash
# Basic usage — language auto-set from .qvoice metadata
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice \
    --text "Ciao, come stai?" -o output.wav

# Works on server (WDELTA decompressed once at startup, zero overhead per request)
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --serve 8080

# Works with instruct for style control (1.7B only)
./qwen_tts -d qwen3-tts-1.7b --load-voice mario_17b.qvoice \
    --text "Una notizia importante." -I "Parla con voce triste" -o sad.wav

# Works with streaming
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice \
    --text "Ciao!" --stdout | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

## Standard Format (Lightweight Alternative)

If you only have the Base model or want smaller files, create a `.qvoice` without
`--target-cv`. The voice is recognizable but prosody may vary slightly from the
original clone when loaded on CustomVoice.

```bash
# Only need the Base model
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio mario.wav -l Italian \
    --voice-name "Mario" --save-voice mario_light.qvoice

# Use on Base model (identical output)
./qwen_tts -d qwen3-tts-0.6b-base --load-voice mario_light.qvoice \
    --text "Ciao, come stai?" -o output.wav

# Use on CustomVoice (voice similar, not identical)
./qwen_tts -d qwen3-tts-0.6b --load-voice mario_light.qvoice \
    --text "Ciao, come stai?" -o output.wav
```

## Comparison: Delta vs Standard

| | Delta | Standard |
|---|---|---|
| **Voice fidelity on CV** | **Bit-identical** to Base clone | Good — voice recognizable, prosody varies |
| **File size (0.6B / 1.7B)** | 785 MB / 2.8 GB | 16 MB / 24 MB |
| **Models needed to create** | Base + CustomVoice | Base only |
| **Models needed to use** | CustomVoice only | CustomVoice or Base |
| **RTF overhead vs preset** | +7% | None |
| **Instruct support (1.7B)** | Yes | Yes |
| **Server support** | Yes | Yes |

**When to use Delta:** Voice accuracy matters — podcasts, audiobooks, production use.

**When to use Standard:** You want a small portable file and "close enough" is fine,
or you only run on the Base model.

## Managing Voice Profiles

These commands don't require a model — they read/manage `.qvoice` files directly.

```bash
# List all .qvoice files in a directory
./qwen_tts --list-voices ./my_voices/

# Inspect a single .qvoice file
./qwen_tts --list-voices my_voice.qvoice

# Delete a voice profile
./qwen_tts --delete-voice ./my_voices/old_voice.qvoice
```

Example output:

```
Voice profiles in my_voices/:
  mario_06b.qvoice     v3    375 frames (30.0s ref)  804134.5 KB  [Mario]  lang=Italian  model=0.6B
  peter_17b.qvoice     v3    375 frames (30.0s ref)  2969485.2 KB  [Peter]   lang=English  model=1.7B
  2 voice profile(s)
```

## `.qvoice` v3 Metadata

Voice files include metadata that prevents common mistakes:

```bash
# Language is auto-set when loading (no need for -l flag)
./qwen_tts -d qwen3-tts-0.6b --load-voice mario_06b.qvoice --text "Ciao!"
#   Auto-set language from voice: Italian

# Warning if you override with wrong language
./qwen_tts -d qwen3-tts-0.6b --load-voice mario_06b.qvoice -l English --text "Hello"
#   WARNING: voice was created with language 'Italian' but you specified 'English'
```

### What's stored in a `.qvoice` v3 file

| Section | Content | Notes |
|---------|---------|-------|
| Header | Version, enc_dim, flags | Compatibility checks |
| Speaker embedding | Float32 vector (1024 or 2048-dim) | From ECAPA-TDNN encoder |
| Reference text | UTF-8 string | Original transcription (if provided) |
| Reference codes | Codec tokens (16 codebooks × N frames) | For in-context learning |
| Metadata | Voice name, language, source model size | Auto-applied at load time |
| Weight deltas (Delta only) | Int16 deltas, LZ4 compressed | Per-tensor, ~785 MB for 0.6B |

### File naming convention

Include the target model size to avoid confusion:
- `mario_06b.qvoice` — for 0.6B CustomVoice
- `mario_17b.qvoice` — for 1.7B CustomVoice

Delta files must match the target CV model exactly.

## Server with Custom Voices

Load a `.qvoice` at server startup — WDELTA decompression happens once, then all
requests use the custom voice with zero overhead.

```bash
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --serve 8080

# Clients don't need to specify language or speaker
curl -s http://localhost:8080/v1/tts -d '{"text":"Ciao!"}' -o out.wav
curl -sN http://localhost:8080/v1/tts/stream -d '{"text":"Ciao!"}' | \
  play -t raw -r 24000 -e signed -b 16 -c 1 -
```

**RTF with custom voice** (Apple M1, 4 threads, Italian, seed 42):

| Mode | 0.6B RTF | 1.7B RTF |
|------|----------|----------|
| CLI | 1.48 | — |
| Server (warm) | **1.44** | 3.32 |
| Server stream | **1.48** | 3.18 |
| Server (cold) | 2.01 | 3.57 |

Custom voices have **no meaningful RTF penalty** compared to preset voices.
The cold/warm gap is OS page cache (first request pages in mmap'd weights from SSD).

> **Note:** Per-request voice switching is not supported. WDELTA weight application
> modifies the model weights in-place — too heavy for hot-swap. Start one server
> per voice if you need multiple custom voices.

## Troubleshooting

**"ERROR: .qvoice enc_dim mismatch"** — The `.qvoice` was created with a different model
size. A file from 0.6B-Base (`enc_dim=1024`) cannot be used on 1.7B and vice versa.
Create separate files per model size (e.g., `mario_06b.qvoice`, `mario_17b.qvoice`).

**"ERROR: WDELTA target_hidden_size mismatch"** — The delta `.qvoice` was created with
`--target-cv` pointing to a different model size. A delta targeting 0.6B cannot be loaded on 1.7B.

**"ERROR: WDELTA voices cannot be loaded on Base models"** — Delta `.qvoice` files must
be loaded on the corresponding CustomVoice model, not on a Base model.

**Voice sounds different than the original clone** — Expected with standard (non-delta)
`.qvoice` on CustomVoice. Use `--target-cv` when creating for bit-identical output.

**Language mismatch warning** — The `.qvoice` stores the language used during creation.
Omit `-l` to use the voice's native language from `.qvoice` metadata.

## Model Compatibility Matrix

| Model Type | Use Case | Voice Source | Style Control | Clone Fidelity |
|------------|----------|-------------|---------------|----------------|
| **Base** | Direct voice clone | `--ref-audio` / `.qvoice` | None | Perfect |
| **CustomVoice** | Preset voices | 9 built-in | `--instruct` (1.7B) | N/A |
| **CustomVoice + .qvoice delta** | Custom cloned voice | `.qvoice` from Base | `--instruct` (1.7B) | **Perfect** (bit-identical) |
| **CustomVoice + .qvoice standard** | Custom cloned voice | `.qvoice` from Base | `--instruct` (1.7B) | Good (prosody varies) |
| **VoiceDesign** | Voice from description | Text description | `--instruct` | N/A |
