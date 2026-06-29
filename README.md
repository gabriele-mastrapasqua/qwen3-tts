# Qwen3-TTS Pure C Implementation

[![Build](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml)
[![CodeQL](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml)
[![Memory Safety](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml)

A lightweight, cross-platform C inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (0.6B and 1.7B). No Python, no PyTorch, no ONNX runtime — just C, a BLAS library, and raw model weights.

The engine runs the complete TTS pipeline: BPE tokenization, a 28-layer causal transformer (Talker), a multi-pass code predictor, and a convolutional speech decoder. Weights are memory-mapped directly from safetensors files in BF16, so loading is near-instant and memory usage stays low.

> 📍 **Where does a voice live in the model?** See **[`docs/speaker-map.md`](docs/speaker-map.md)** for a readable map of which layers/stages carry timbre vs language/prosody vs emotion (and how the preset voices like `ryan` work). Essential background for voice cloning and expressivity.

## Audio Samples

All samples generated with the 0.6B model (RTF ~1.3–1.7, Apple M1):

| Language | Speaker | Sample | Text |
|----------|---------|--------|------|
| English | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/english_ryan.wav) | *Hello, this is a test of the text to speech system.* |
| Italian | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/italian_ryan.wav) | *Buongiorno a tutti, questa e una dimostrazione del sistema di sintesi vocale.* |
| Italian | vivian | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/italian_vivian.wav) | *Buongiorno a tutti, questa e una dimostrazione del sistema di sintesi vocale.* |
| Spanish | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/spanish_ryan.wav) | *Hola, esta es una demostracion del sistema de sintesis de voz.* |
| Portuguese | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/portuguese_ryan.wav) | *Ola, esta e uma demonstracao do sistema de sintese de voz.* |
| French | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/french_ryan.wav) | *Bonjour a tous, ceci est une demonstration du systeme de synthese vocale.* |
| German | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/german_ryan.wav) | *Guten Tag, dies ist eine Demonstration des Sprachsynthesesystems.* |
| Japanese | Ono_Anna | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/japanese_ono_anna.wav) | *こんにちは、私の名前はアンナです。今日はとても良い天気ですね。東京の桜がとても綺麗です。* |
| Japanese | Ono_Anna | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/ganbatte_andrea.wav) | *頑張れ、アンドレア！あなたならできるよ。毎日少しずつ前に進もう。夢を諦めないで。応援してるよ！* |

> Clone and play locally: `afplay samples/english_ryan.wav` (macOS) or `aplay samples/english_ryan.wav` (Linux)

## Quick Start

```bash
# Clone and build
git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts
make blas

# Download a model (interactive: small, large, voice-design, base-small, base-large)
./download_model.sh

# Synthesize speech
./qwen_tts -d qwen3-tts-0.6b --text "Hello, how are you today?" -o hello.wav
```

> **Dependencies:** Only a C compiler and BLAS (Accelerate on macOS, OpenBLAS on Linux).
> See [docs/building.md](docs/building.md) for Linux, Windows/WSL2, and other build targets.

## Features

- **Pure C, minimal dependencies** — Only requires a C compiler and BLAS. No Python runtime needed.
- **Runs on macOS, Linux and Windows/WSL2 (ARM/x86)** — the hot matvec/attention kernels have **NEON+SDOT (ARM), AVX2 and AVX-512/VNNI (x86)** twins with a scalar fallback + runtime ISA guard, and decode threading runs on a **cross-OS pool** (GCD on macOS, pthread elsewhere). Validated on Apple M1, Ryzen 7 6800H, and EPYC 9555P (Zen5). Single-stream RTF is memory/cache-bound, so the chip's cache matters most (see [Performance](#performance)); measure yours with `bash tests/x86_bench.sh`.
- **Both model sizes** — Automatically detects 0.6B or 1.7B from weight files.
- **9 preset voices** — `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.
- **10 languages** — English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Memory-mapped weights** — BF16 safetensors mmap'd directly. 0.6B ~3 GB, 1.7B ~8 GB.
- **Voice cloning** — Clone any voice from a short WAV clip. Ship it as a compact **~25 MB graft `.qvoice`** (`tests/qvoice_to_graft.py` → `--icl-only`): keeps the CustomVoice weights so emotion levers (`--instruct`, `--expr`, `--ml-steer`) all work, with full prosody (sighs/pauses). An 8 KB `--xvector-only` `.bin` is the ultra-lean alternative (identity only). See `docs/icl-graft-portability.md`.
- **Voice management** — List, inspect, delete `.qvoice` profiles (`--list-voices`, `--delete-voice`). No model required.
- **Style control** — `--instruct` for emotion/style on 1.7B: angry, whisper, cheerful, and more.
- **Expressivity presets** — `--emotion <mood>` drives a full ear-validated recipe at once (vector + steer-weight + roughness + volume + rate): compound moods `joy, sad, annoyed, stern, angry` plus the base palette `happy, excited, eager, proud, sad, gloomy, news, dramatic, calm`, for controllable delivery where `--instruct` is weak. Language-aware (Italian uses the centered palette), blendable (`happy:0.5,proud:0.5`), dial with `--steer-weight`, plus a `--roughness` grit knob and pitch-preserving `--rate`/`--volume`. Cross-model, works on custom voices. See [docs/expressivity.md](docs/expressivity.md).
- **Inline markup for audiobooks** — write one text with ElevenLabs/Bark-style tags and get a multi-emotion take in one pass: `--text "I won! [excited] ...amazing! [pause:500ms] [sad] But it's over. [sigh]"`. Mid-text emotion switches, `[pause:400ms]`/`[break:1s]` pauses, and `[sigh]`/`[huff]` paralinguistic fillers — auto-detected in `--text` (no flag) or explicit via `--compose`. Spans are model-generated and concatenated seamlessly. See [docs/markup.md](docs/markup.md).
- **VoiceDesign** — Create new voices from text descriptions.
- **HTTP server** — `/v1/tts`, `/v1/tts/stream`, OpenAI-compatible `/v1/audio/speech`.
- **Streaming** — Real-time audio via `--stream` (WAV) or `--stdout` (raw PCM).
- **INT8 quantization** — `--int8` quantizes Talker + Code Predictor (native SDOT on ARM, AVX-512/VNNI on x86): **0.6B goes sub-realtime on Apple Silicon (RTF < 1.0, CLI/stream/server)**, **1.7B 2.66→1.79 (−33%)**, near-bf16 quality, works with preset speakers and custom `.qvoice` voices. (INT4 is the lever on memory-starved x86; on cache-rich chips like M1, INT8 wins.)
- **Configurable sampling** — Temperature, top-k, top-p, and repetition penalty.
- **24 kHz WAV output** — 16-bit PCM, mono.

## Usage

```
./qwen_tts [options]

Required:
  -d, --model-dir <path>     Model directory
  --text <string>            Text to synthesize

Optional:
  -o, --output <path>        Output WAV file (default: output.wav)
  -s, --speaker <name>       Speaker voice (default: ryan)
  -l, --language <lang>      Target language (default: English)
  -I, --instruct <text>      Style/emotion instruction (1.7B model only)
  --temperature <f>          Sampling temperature (default: 0.5)
  --top-k <n>                Top-k sampling (default: 50)
  --top-p <f>                Top-p nucleus sampling (default: 1.0)
  --rep-penalty <f>          Repetition penalty (default: 1.05)
  --max-tokens <n>           Max audio tokens (default: 8192)
  --max-duration <secs>      Max audio duration in seconds
  --seed <n>                 Random seed for reproducible output
  --ref-audio <path>         Reference audio for voice cloning (Base model)
  --save-voice <path>        Save voice profile (.qvoice = full, .bin = x-vector only)
  --load-voice <path>        Load voice profile (.qvoice or .bin)
  --xvector-only             Clone via speaker x-vector only — clean, 8KB .bin (recommended for expr/emotion)
  --icl-only                 Graft mode: keep CV weights, use the .qvoice ICL prefix (max timbre mimicry)
  --target-cv <dir>          CV model dir for delta encoding (bit-identical cross-model)
  --list-voices <dir>        List .qvoice files in directory (no model needed)
  --delete-voice <path>      Delete a .qvoice file
  --voice-name <name>        Name for the voice (stored in .qvoice metadata)
  --voice-design             VoiceDesign mode (create voice from --instruct)
  --stream                   Stream audio (decode chunks during generation)
  --stdout                   Output raw s16le PCM to stdout (implies --stream)
  --int8                     INT8 quantized (0.6B & 1.7B; faster, ~same quality) — recommended; uses VNNI on AVX-512 x86, SDOT on ARM
  --int4                     Q4_0 quantized (experimental; slower than --int8 on CPU)
  -j, --threads <n>          Worker threads (default: 4)
  --silent                   Suppress status output
  --debug                    Verbose diagnostics
  --serve <port>             Start HTTP server
```

### Examples

```bash
# Basic English
./qwen_tts -d qwen3-tts-0.6b --text "The quick brown fox jumps over the lazy dog." -o fox.wav

# Italian with a specific voice
./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian \
    --text "Ciao, questa e una prova del sistema di sintesi vocale." -o test_it.wav

# Style/emotion control (1.7B only)
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
    --text "I cannot believe you did that to me." \
    --instruct "Speak in a very angry and aggressive tone" -o angry.wav

# Reproducible output with seed
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --seed 42 -o hello.wav
```

### Voice Cloning

Clone any voice from a reference audio clip. Requires a Base model.

```bash
# Clone a voice
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio reference.wav \
    --text "Hello, this is my cloned voice." -o cloned.wav
```

> Full guide: reference audio tips, model comparison, samples → [docs/voice-cloning.md](docs/voice-cloning.md)

### Custom Voices with Delta `.qvoice`

The killer feature: clone a voice once, save it as a `.qvoice` with `--target-cv`,
and use it forever on the CustomVoice model — **bit-identical** to the original clone.
Works with `--instruct`, streaming, and the HTTP server.

```bash
# Create (one-time: needs both Base + CV models)
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio mario.wav -l Italian \
    --voice-name "Mario" --target-cv qwen3-tts-0.6b \
    --save-voice mario.qvoice

# Use forever (only CV model + .qvoice needed)
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice \
    --text "Ciao, come stai?" -o output.wav

# On the server
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --serve 8080

# Manage voice profiles (no model needed)
./qwen_tts --list-voices ./my_voices/
./qwen_tts --delete-voice ./my_voices/old.qvoice
```

**Voice clone samples** — all generated via `.qvoice` delta on 0.6B CustomVoice:

| Language | Voice | Source | Output | Text |
|----------|-------|--------|--------|------|
| Italian | Pirandello Reader | [LibriVox](https://librivox.org/) Public Domain | [input](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/ref_italian_pirandello.wav) → [clone](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_italian_06b.wav) | *Buongiorno a tutti, questa e una dimostrazione della clonazione vocale.* |
| English | Sarac (F) | [LibriTTS-R](https://www.openslr.org/141/) CC-BY | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_sarac_english_06b.wav) | *Good morning everyone, this is a demonstration of voice cloning using a custom voice profile.* |
| English | Peter (M) | [LibriTTS-R](https://www.openslr.org/141/) CC-BY | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_peter_english_06b.wav) | *I love reading books aloud, there is something magical about bringing stories to life with your voice.* |
| French | Baudelaire Reader | [LibriVox](https://librivox.org/) Public Domain | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_french_06b.wav) | *Bonjour a tous, ceci est une demonstration du clonage vocal avec un profil de voix personnalise.* |
| Spanish | Lu | [LibriVox](https://librivox.org/) Public Domain | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_spanish_06b.wav) | *Buenos dias a todos, esta es una demostracion de la clonacion de voz con un perfil de voz personalizado.* |

> Full guide: delta vs standard, format internals, troubleshooting → [docs/custom-voices.md](docs/custom-voices.md)

### Emotion & expressivity (1.7B)

Emotion is **one flag**. Pick an emotion with `--emotion` and the engine auto-composes the validated
**COMBINE** stack for you — the per-language fine-tune (`.expr`) **plus** the steering vector for that
voice and emotion, at the ear-validated weights. No file paths, no layer ranges. A vivid **English or
Chinese** `--instruct` on top is optional but **recommended** — it drives the strongest, most natural result.

```bash
# emotion in ONE flag — works on presets AND cloned voices
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 --emotion sad \
    --instruct "Speak softly, with quiet sadness." \
    --text "Allora, lascia che ti spieghi come stanno le cose." -o sad.wav
```

- **Emotions:** `sad · joy · anger · fear · disgust · surprise` (synonyms like `happy`/`angry` work too).
- **Works in every Qwen3-TTS language** (EN, IT, DE, ZH, RU, KO, JA, ES, FR, PT) — just set `-l <Language>`.
  The steering vector carries the emotion (a language-agnostic activation direction); the `.expr` renders and
  stabilizes the language. **DE / FR / IT have a native `.expr`**; the other languages use the Italian pack as
  a universal cross-language renderer (ear-validated on IT/ES/RU/ZH/JA/KO).
- **Cloned voices** emote with the same one flag (the engine picks the clone's own steer dir when it has one,
  else a universal palette). Clones default to a clean **x-vector** load; see [Voice Cloning](#voice-cloning).
- **Paralinguistics → inline `[tags]`, also automatic.** Write `[laugh]` or `[sigh]` in `--text` and the engine
  performs the event (it picks the onomatopoeia anchor + the right vector for you) — no flags:
  ```bash
  ./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
      --text "Che giornata... [sigh] non ce la faccio più. [laugh]" -o para.wav
  ```

**Get the assets** (needed once, for the `.expr` fine-tunes — the steering vectors already ship in this repo):

```bash
bash download_assets.sh            # fetch the .expr packs into presets/expr/ (sha256-verified)
bash download_assets.sh --verify   # re-check integrity of what you already have
```
They are hosted on Hugging Face → [**gabrione/qwen3-tts-italian-expr**](https://huggingface.co/gabrione/qwen3-tts-italian-expr)
(one repo, folders by purpose). The full set ≈ 1.4 GB; Italian-only emotion needs just `italian_csp_topk6.expr` (203 MB).

<details>
<summary><b>Under the hood</b> — what each file is, and the manual override flags (advanced)</summary>

You normally never touch these — `--emotion` and `[tags]` load the right ones. But if you want to tune by hand:

| File | Where | What it is | Manual flag |
|------|-------|------------|-------------|
| `.expr` | `presets/expr/` (HF) | **Per-language emotion fine-tune** — a weight-delta on the Talker's emotion layers; fixes/renders the language *and* gives the base emotion. | `--expr <file> --expr-weight <m>` |
| `.qlsteer` | `presets/steer/emotion/` (git) | **Emotion steering vector** — an inference-time activation direction (per voice × emotion). Changes no weights; carries the emotion, transfers cross-voice/language. | `--ml-steer <file> --ml-range 21-25 --ml-weight <w>` |
| `.qlsteer` | `presets/steer/paraling/` (git) | **Paralinguistic vector** — `laugh_vs_cry`, `sigh_vs_laugh`. Speaker/language-agnostic. | (auto via `[laugh]`/`[sigh]`) |
| `.qamp` | `presets/steer/paraling/` (git) | **Raw activation fingerprint** — the source a `.qlsteer` is built from (reproducibility). | (build input) |

A manual `--expr` / `--ml-steer` always **overrides** the `--emotion` auto-router. Validated tuning notes:
steer `w8` sweet spot (`w12` over-steers); ryan-anger uses expr only (steer goes metallic); clones take
`--expr-weight ~1.2`. **Train your own `.expr` for any language** with [`training/expressivity-lora/`](training/expressivity-lora/).
</details>

**Deeper docs:** [docs/expressivity-assets.md](docs/expressivity-assets.md) (asset catalog + recipes) ·
[docs/csp-ft-emotion.md](docs/csp-ft-emotion.md) (how the `.expr` packs were trained, cross-language transfer) ·
[docs/expressivity-lora.md](docs/expressivity-lora.md) (which layers, the `.expr` format, train your own) ·
[docs/paralinguistics-tags.md](docs/paralinguistics-tags.md) (laugh/sigh tags + vectors).

### HTTP Server

```bash
# Start server
./qwen_tts -d qwen3-tts-0.6b --serve 8080

# Serve many users at once — step their requests together (vLLM-style request batching)
./qwen_tts -d qwen3-tts-0.6b --serve 8080 --batch-size 4

# Generate speech
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"Hello, how are you?"}' -o output.wav

# Stream with real-time playback
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you?"}' | \
  play -t raw -r 24000 -e signed -b 16 -c 1 -

# OpenAI-compatible endpoint
curl -s http://localhost:8080/v1/audio/speech \
  -d '{"input":"Hello world","voice":"ryan"}' -o output.wav
```

> Full guide: all endpoints, request body, performance → [docs/server.md](docs/server.md)

### Streaming

```bash
# Stream to WAV file
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stream -o hello.wav

# Pipe raw PCM to audio player
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stdout | \
    play -t raw -r 24000 -e signed -b 16 -c 1 -
```

## How It Works

```
Text --> BPE Tokenizer --> Talker (LLM) --> Code Predictor --> Speech Decoder --> 24 kHz WAV
```

| Component | What it does |
|-----------|-------------|
| **Talker** | 28-layer Qwen3 transformer with GQA, RoPE, SwiGLU. Generates one audio frame token per step. |
| **Code Predictor** | 5-layer transformer running 15 sequential passes per frame. Predicts the remaining 15 codebook entries. |
| **Speech Decoder** | Causal ConvNet with 16-codebook RVQ dequantization and 480x upsampling. Converts codes to waveform. |

| | 0.6B | 1.7B |
|---|------|------|
| Talker hidden dim | 1024 | 2048 |
| Heads (Q/KV) | 16/8 | 16/8 |
| Layers | 28 | 28 |
| Code Predictor | 1024 hidden, 5 layers | 1024 hidden, 5 layers (+2048→1024 projection) |
| Memory | ~3 GB | ~8 GB |

## Performance

> ### ⚡ The sweet spot: `--int8` is **faster than real-time (RTF < 1.0) on Apple Silicon** — CLI, streaming **and** server
> ~2× faster than bf16 with **no perceptible quality loss** (validated by ear, including cloned `.qvoice` voices).

**Apple M1** (8-core, 16 GB, 4 threads), 0.6B model — full-precision **bf16** vs **`--int8`**, across every delivery mode:

| Mode | bf16 RTF | **`--int8` RTF** | First audio (TTFA) |
|---|---|---|---|
| CLI (short, ~4 s) | 1.5–1.8 | **0.90** ⚡ | 0.96 s |
| CLI (long, ~14 s) | ~1.3 | **0.80** ⚡ | — |
| **Streaming** (`--stream`, short) | 1.5–1.8 | **0.89** ⚡ | **0.46 s** |
| **Streaming** (long) | ~1.3 | **0.81** ⚡ | **0.50 s** |
| **HTTP server** (`--serve`, warm) | ~1.3 | **0.88** ⚡ | — |
| **Custom voice** `.qvoice` (streamed) | 1.34 | **0.93** ⚡ | 0.47 s |

Yes — this project ships a **streaming mode** (`--stream`, ~0.5 s to first audio) and an
**OpenAI-compatible HTTP server** (`--serve`, with `--workers N` request concurrency). With `--int8`,
**every delivery path runs faster than real time on a 2020 M1** — cloned custom voices included.

RTF = processing_time / audio_duration; **< 1.0 = faster than real-time**. `--int8` quantizes the
Talker + Code Predictor (native SDOT on ARM, AVX-512/VNNI on x86): **0.6B drops from ~1.5 (bf16) to
~0.8–0.9**, **1.7B 2.66 → 1.79 (−33%)**, no perceptible quality loss, and it works with `.qvoice`
voices ([details](docs/quantization.md)). 1.7B: bf16 ~2.0–4.1, `--int8` ~1.8–2.4 on longer text.

### 📊 Benchmark *your* CPU

Want to know how this runs on **your** machine (Apple Silicon, AMD/Intel x86, ARM server)? The repo
ships a one-command per-box report — no setup beyond the model:

```bash
make bench              # quick RTF: short+long, normal+stream (both models)
make bench-full         # + server, instruct, INT8, .qvoice

# Per-CPU report (copy onto any rented ARM/x86 box):
./qwen_tts --caps       # what SIMD your CPU actually has (NEON/SDOT/bf16/i8mm/SVE • AVX2/AVX-512/VNNI/AMX)
./qwen_tts --self-test  # are the kernels numerically correct on this ISA?
make bench-matrix       # caps + self-test + RTF matrix (single vs batch × bf16/int8/int4)
make bench-matrix-full  # + streaming + server + request-batching throughput
make bench-server       # concurrent-request throughput alone (N users vs single-stream, per precision)
```

The full cross-hardware workflow (which boxes have which SIMD, where to rent, what to measure) lives in
[docs/hardware-testing.md](docs/hardware-testing.md).

**Cross-device CPU (single-stream 0.6B, this repo's best config — reproduce with `bash tests/x86_bench.sh`):**

| Device | SIMD + threads | RAM | Best 0.6B RTF | Config |
|---|---|---|---|---|
| **Apple M1** 8-core | NEON + SDOT int8, GCD 4-thread | 16 GB | **~1.3 bf16 / sub-1.0 int8** | `--int8 -j4` |
| **Ryzen 7 6800H** (Zen3+, 16 MB L3, bare metal) | AVX2 + FMA, pthread 4-thread | 32 GB | **2.02** | `--int4 -j4` |
| **EPYC 9555P** (Zen5, AVX-512+VNNI, Scaleway VM) | AVX-512-VNNI, pthread | 16 GB / 4 vCPU | **1.64** | `--int8 -j1` |

Single-stream RTF is **memory/cache-bound** (the Code Predictor re-reads its weights 16×/frame):
SIMD width and thread count matter less than fewer weight bytes (`--int8`/`--int4`) and a cache
that fits the working set (Apple's SLC, an X3D chip's V-cache). On x86 the int8+VNNI kernel stack
is a real **~1.85× win at equal core count** (EPYC 9555P: scalar-bf16 `-j1` 3.04 → VNNI-int8 `-j1`
1.64); threading scales on bare metal but a multi-CCD VM limits it. Many-core servers are best for
**throughput** (concurrent requests), not single-stream latency. Check yours: `./qwen_tts --caps`.

**Concurrent serving — request batching (`--serve --batch-size N`).** For *N users at once*, the server
can step their requests **together** through the model (vLLM-style): weights are read from memory **once**
and reused across all in-flight requests, instead of re-read per user. A continuous scheduler keeps the
batch full (a finished request's slot is refilled immediately) and **streaming composes** — each user
still gets their own progressive audio stream. This trades a little per-request latency for much higher
total throughput on bandwidth-bound boxes. Measure it on your CPU with `make bench-server`; details in
[docs/server-batching.md](docs/server-batching.md).

**vs other implementations:**

| Hardware | 0.6B RTF | Notes |
|----------|----------|-------|
| **This project (C, Apple M1 CPU, `--int8`)** | **0.80–0.90** | Pure C, no GPU — **faster than real-time** |
| This project (C, Apple M1 CPU, bf16) | 1.26–1.39 | Pure C, no GPU |
| Python + PyTorch (Ryzen 9 7950X CPU) | 4.5–5.8 | Official Python, CPU-only |
| NVIDIA RTX 3090 | 0.52–0.68 | Python + PyTorch + FlashAttention 2 |

5–7x faster than Python on CPU, and **faster than real-time with `--int8`** — on a 2020 laptop with no GPU.

> Per-component breakdown, full GPU table, optimization history → [docs/performance.md](docs/performance.md)
> x86 AVX2/AVX-512/VNNI findings + how to benchmark your CPU → [docs/x86-optimization.md](docs/x86-optimization.md)

## Documentation

| Guide | Contents |
|-------|----------|
| [Voice Cloning](docs/voice-cloning.md) | Reference audio tips, ECAPA-TDNN internals, model comparison, samples |
| [Custom Voices](docs/custom-voices.md) | `.qvoice` format, delta vs standard, managing profiles, troubleshooting |
| [HTTP Server](docs/server.md) | All endpoints, request body, streaming, server performance |
| [Server request-batching](docs/server-batching.md) | vLLM-style `--batch-size N`: serve N concurrent users together, continuous batching, per-request streaming |
| [VoiceDesign](docs/voice-design.md) | Creating voices from text descriptions |
| [Expressivity](docs/expressivity.md) | `--emotion` compound moods + presets, mood blending, `--roughness`, `--rate`/`--volume`, building your own control vectors |
| [Expressivity packs `.expr`](docs/expressivity-lora.md) | Per-language emotion LoRA: which layers, why it's ~16–63 MB, file format, `--expr`/`--expr-weight`, per-voice rank. Train your own: [`training/expressivity-lora/`](training/expressivity-lora/) |
| [Inline markup](docs/markup.md) | Audiobook/podcast tags in `--text`: `[sad]`/`[excited]` mid-text emotion switches, `[sigh]`/`[huff]` fillers, `[pause:400ms]` |
| [Quantization](docs/quantization.md) | INT8/INT4, comparison table, recommendations |
| [Performance](docs/performance.md) | RTF benchmarks, component breakdown, CPU vs GPU, optimization history |
| [x86 optimization](docs/x86-optimization.md) | AVX2 / AVX-512 / VNNI findings, why it's memory-bound, how to benchmark your CPU |
| [Hardware testing / benchmark your CPU](docs/hardware-testing.md) | One-command per-box report (`make bench-matrix`), which CPU has which SIMD, where to rent ARM/x86, the RTF + throughput matrix to fill in |
| [Building](docs/building.md) | All platforms, build targets, testing (golden-reference test needs Python + `librosa`) |

### Blog Posts

| Post | Topic |
|------|-------|
| [Voice Cloning Internals](blog/voice-cloning-internals.md) | ECAPA-TDNN architecture deep-dive |
| [Cross-Model Voice Analysis](blog/cross-model-voice-analysis.md) | Why delta format works (weight analysis) |
| [Optimization Notes](blog/optimization-notes.md) | RTF 3.5 → 1.3: the full M1 bf16 optimization story |
| [Fast on Every CPU](blog/making-qwen3-tts-fast-on-every-cpu.md) | SDOT (sub-1.0 on M1) + AVX2/AVX-512/VNNI on x86; why it's memory-bound |

## Credits & Acknowledgments

- **Salvatore Sanfilippo ([antirez](https://github.com/antirez))** — This project wouldn't exist without his [qwen-asr](https://github.com/antirez/qwen-asr), a pure C Qwen2-Audio ASR engine that proved you can do real neural inference in plain C with mmap'd safetensors, BF16 NEON kernels, and zero dependencies. The entire architecture of this TTS engine — the approach, the style, the philosophy of minimal C inference — is directly inspired by his work. If you like this project, go star qwen-asr first.
- **Michael Abrash** — His *[Graphics Programming Black Book](https://www.jagregory.com/abrash-black-book/)* (1997) shaped how we think about performance. The chapters on data alignment, struct layout, and cache-friendly access patterns for the 386/486 are still relevant today — we got a **24% speedup** from cache-line alignment (`posix_memalign(64)`), applying the same principles Abrash taught 30 years ago to modern SIMD and BLAS.
- **John Carmack** — His `.plan` files and QuakeCon talks on micro-optimization and cache friendliness were a constant reference. Where Abrash gave you the systematic rules and benchmarks, Carmack showed you the mindset: always think about how data flows through the CPU.
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team at Alibaba — the model architecture, weights, and research. Models on [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). [Paper](https://arxiv.org/abs/2505.15894).
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team — the base LLM architecture (GQA, RoPE, SwiGLU) used in the Talker and Code Predictor.

## License

MIT
