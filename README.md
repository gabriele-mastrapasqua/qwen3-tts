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
- **Optional GPU backends (opt-in)** — **Apple Metal** (`make metal`) and **NVIDIA CUDA** (`make cuda`) run the whole fused pipeline resident on the GPU (~0.36 RTF for 0.6B on an M2 Pro; ~0.44 for 1.7B on a mainstream NVIDIA GPU), plus server request-batching for throughput. CPU stays the default. → [Performance § GPU backends](#performance) · [docs/hardware-testing.md](docs/hardware-testing.md) (Metal) · [docs/cuda-performance.md](docs/cuda-performance.md) (CUDA).
- **Both model sizes** — Automatically detects 0.6B or 1.7B from weight files.
- **9 preset voices** — `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.
- **10 languages** — English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Memory-mapped weights** — BF16 safetensors mmap'd directly. 0.6B ~3 GB, 1.7B ~8 GB.
- **Voice cloning** — Clone any voice from a short WAV clip. Ship it as a compact **~25 MB graft `.qvoice`** (`tests/qvoice_to_graft.py` → `--icl-only`): keeps the CustomVoice weights so emotion levers (`--instruct`, `--expr`, `--ml-steer`) all work, with full prosody (sighs/pauses). An 8 KB `--xvector-only` `.bin` is the ultra-lean alternative (identity only). See `docs/icl-graft-portability.md`.
- **Voice management** — List, inspect, delete `.qvoice` profiles (`--list-voices`, `--delete-voice`). No model required.
- **Style control** — `--instruct` for emotion/style on 1.7B: angry, whisper, cheerful, and more.
- **Emotion in one flag** (🧪 **beta**; paralinguistics `[laugh]`/`[sigh]` 🧪 **alpha**) — `--emotion <sad\|joy\|anger\|fear\|disgust\|surprise>` (1.7B) auto-applies the ear-validated recipe (per-language fine-tune `.expr` + steering vector + a default English instruct + temperature), on presets **and** cloned voices, in every Qwen language. **Plus 7 blended "dyads"** (`contempt`, `awe`, `nostalgia`, `disapproval`, `remorse`, `outrage`, `despair`) and **inline `[emotion]` switching** — many emotions from one prompt in a single generation. A vivid English `--instruct` and `-T` override. Pitch-preserving `--rate`/`--volume` and a `--roughness` grit knob are still available. See [docs/emotion-THE-recipe.md](docs/emotion-THE-recipe.md).
- **Inline markup for audiobooks** — write one text with ElevenLabs/Bark-style tags and get a multi-emotion take in one pass: `--text "I won! [excited] ...amazing! [pause:500ms] [sad] But it's over. [sigh]"`. Mid-text emotion switches, `[pause:400ms]`/`[break:1s]` pauses, and `[sigh]`/`[huff]` paralinguistic fillers — auto-detected in `--text` (no flag) or explicit via `--compose`. Spans are model-generated and concatenated seamlessly. See [docs/markup.md](docs/markup.md).
- **VoiceDesign** — Create new voices from text descriptions.
- **HTTP server** — `/v1/tts`, `/v1/tts/stream`, OpenAI-compatible `/v1/audio/speech`; JSON body takes `emotion`/`instruct`/`volume`/`rate` (same recipe as the CLI). **Inline `[mood]` markup works over the API too** — one request can switch emotion sentence-by-sentence (`"text":"[joy] Great news! [sad] But I must go."`), auto-detected and streamed span-by-span. See [docs/server.md](docs/server.md).
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

**Ready-to-use reference voices (CC0 / Public Domain).** Four lite ~25 MB graft `.qvoice` clones of LibriVox
public-domain readers (Italian, Spanish, English, French) so the demos/tests run out of the box and you have
voices to listen to and reuse:
```bash
bash download_voices.sh    # fetch galatea(IT)/quijote(ES)/ohenry(EN)/hugo(FR) into voices/ (sha256-verified)
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_graft.qvoice --icl-only -l Italian \
    --text "Buongiorno, questa è la mia voce clonata." -o out.wav
```
> 🎭 Want these clones to **emote** (`--emotion`)? See the **Emotion & expressivity** section below — it needs `bash download_assets.sh` first.
Hosted on Hugging Face → [**gabrione/qwen3-tts-voices**](https://huggingface.co/gabrione/qwen3-tts-voices) (CC0, LibriVox attribution).

### Custom Voices — small, portable, and *emotable* `.qvoice`

Clone a voice once, save it as a portable `.qvoice`, reuse it forever on the CustomVoice model — with
`--instruct`, `--emotion`, streaming, and the HTTP server.

**The default `.qvoice` is now a ~25 MB "graft"** — it keeps the CustomVoice weights, so it stays small,
carries full prosody, **and the emotion / instruct levers still work on your clone** (no more multi-GB
weight-delta files):

```bash
# Create — default = ~25 MB graft (one-time; needs the Base model)
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio mario.wav -l Italian \
    --voice-name "Mario" --save-voice mario.qvoice

# Use it on CustomVoice — --icl-only keeps the CV weights (→ instruct/emotion work)
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --icl-only \
    --text "Ciao, come stai?" -o output.wav

# ...with an emotion, on your OWN cloned voice:
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --icl-only \
    --emotion joy -l Italian --text "Ce l'abbiamo fatta!" -o joy.wav

# Server / manage
./qwen_tts -d qwen3-tts-0.6b --load-voice mario.qvoice --icl-only --serve 8080
./qwen_tts --list-voices ./my_voices/
```

**Other formats** → [docs/custom-voices.md](docs/custom-voices.md): **8 KB `.bin` x-vector** (`--xvector-only`,
tiniest & cleanest) · **heavy WDELTA** (`--target-cv`, ~0.8–3 GB, bit-identical — only if you need exact fidelity).

**Voice clone samples** — cloned voices on 0.6B CustomVoice (25 MB grafts):

| Language | Voice | Source | Output | Text |
|----------|-------|--------|--------|------|
| Italian | Pirandello Reader | [LibriVox](https://librivox.org/) Public Domain | [input](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/ref_italian_pirandello.wav) → [clone](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_italian_06b.wav) | *Buongiorno a tutti, questa e una dimostrazione della clonazione vocale.* |
| English | Sarac (F) | [LibriTTS-R](https://www.openslr.org/141/) CC-BY | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_sarac_english_06b.wav) | *Good morning everyone, this is a demonstration of voice cloning using a custom voice profile.* |
| English | Peter (M) | [LibriTTS-R](https://www.openslr.org/141/) CC-BY | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_peter_english_06b.wav) | *I love reading books aloud, there is something magical about bringing stories to life with your voice.* |
| French | Baudelaire Reader | [LibriVox](https://librivox.org/) Public Domain | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_french_06b.wav) | *Bonjour a tous, ceci est une demonstration du clonage vocal avec un profil de voix personnalise.* |
| Spanish | Lu | [LibriVox](https://librivox.org/) Public Domain | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/clone_spanish_06b.wav) | *Buenos dias a todos, esta es una demostracion de la clonacion de voz con un perfil de voz personalizado.* |

> Full guide: delta vs standard, format internals, troubleshooting → [docs/custom-voices.md](docs/custom-voices.md)

### Emotion & expressivity (1.7B) · 🧪 Beta

> 🧪 **Beta quality.** Ear-validated after ~a month of tuning, but results vary by language/voice and can still
> be imprecise — expect rough edges (please gauge that before filing issues 🙏). Improvements will come, just
> not today.
>
> ⚙️ **Setup (once):** run **`bash download_assets.sh`** to fetch the emotion fine-tunes (`.expr`, ~200 MB for
> Italian) from Hugging Face → [**gabrione/qwen3-tts-italian-expr**](https://huggingface.co/gabrione/qwen3-tts-italian-expr).
> The steering vectors already ship in this repo. **`--emotion` then works on the 9 presets AND on your own
> cloned voices.** No clone yet? Grab ready-made **CC0 graft voices** with `bash download_voices.sh` →
> [**gabrione/qwen3-tts-voices**](https://huggingface.co/gabrione/qwen3-tts-voices) and emote them straight away.
>
> 🎧 **Hear it in one command** (after the two downloads): **`make emotion-demo`** renders a batch of emotion
> clips — every language × emotion, on presets *and* the galatea clone — so you can judge the current quality by
> ear. **`make emotion-para-demo`** adds the alpha `[laugh]`/`[sigh]`.

Emotion is **one flag**. Pick an emotion with `--emotion` and the engine auto-composes the validated
**COMBINE** stack for you — the per-language fine-tune (`.expr`) **plus** the steering vector for that
voice and emotion, at the ear-validated weights. No file paths, no layer ranges. A vivid **English or
Chinese** `--instruct` on top is optional but **recommended** — it drives the strongest, most natural result.

```bash
# emotion on a CLONED voice (galatea = a ready-made CC0 graft) — same one flag
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_graft.qvoice --icl-only \
    -l Italian --emotion sad --text "Ho perso tutto, e adesso non so più cosa fare." -o sad.wav
```

```bash
# emotion in ONE flag — works on presets AND cloned voices
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 --emotion sad \
    --instruct "Speak softly, with quiet sadness." \
    --text "Allora, lascia che ti spieghi come stanno le cose." -o sad.wav
```

**🔊 Hear it** — committed examples in [`samples/emotion_examples/`](samples/emotion_examples) (play after clone:
`afplay samples/emotion_examples/<file>.wav`, or click to download):

| Language | Voice | Emotion | Text | Listen |
|----------|-------|---------|------|--------|
| Italian | ryan (preset) | 😢 sad | *Ho perso tutto quello che avevo, e adesso non so più cosa fare.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_sad.wav) |
| Italian | ryan (preset) | 😄 joy | *Non ci posso credere, è la notizia più bella della mia vita!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_joy.wav) |
| Italian | ryan (preset) | 🤢 disgust | *Ma che roba è questa? Fa davvero schifo, non riesco neanche a guardarla.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_disgust.wav) |
| Italian | ryan (preset) | 😲 surprise | *Cosa?! Non me lo aspettavo per niente, è incredibile!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_surprise.wav) |
| Italian | galatea (cloned voice) | 😢 sad | *Ho perso tutto quello che avevo, e adesso non so più cosa fare.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/galatea_it_sad.wav) |
| Italian | galatea (cloned voice) | 😠 anger | *Come ti permetti di parlarmi così? Questo non lo accetto!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/galatea_it_anger.wav) |
| English | ryan (preset) | 😢 sad | *I've lost everything I had, and now I don't know what to do anymore.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_en_sad.wav) |
| English | ryan (preset) | 😠 anger | *How dare you talk to me like that? I will not accept this!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_en_anger.wav) |
| English | ryan (preset) | 😨 fear | *There's someone in the house, I heard footsteps... I'm so scared, I don't know what to do.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_en_fear.wav) |
| German | vivian | 😠 anger | *Also, lass mich dir in Ruhe erklären, wie die Dinge wirklich stehen.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/de_vivian_anger.wav) |
| French | vivian | 😢 sad | *Bon, laisse-moi t'expliquer calmement comment les choses se passent.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/fr_vivian_sad.wav) |
| French | vivian | 😲 surprise | *Quoi ? Je ne m'y attendais pas du tout, c'est incroyable.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/fr_vivian_surprise.wav) |
| Spanish | vivian | 😄 joy | *Bueno, déjame explicarte con calma cómo están realmente las cosas.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/es_vivian_joy.wav) |
| Spanish | vivian | 🤢 disgust | *¿Pero qué es esto? Es asqueroso, ni siquiera puedo mirarlo.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/es_vivian_disgust.wav) |
| Portuguese | ryan | 😄 joy | *Não acredito, é a melhor notícia da minha vida!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/pt_ryan_joy.wav) |
| Chinese | vivian | 😄 joy | *我简直不敢相信，这是我一生中最好的消息，我太高兴了！* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/zh_vivian_joy.wav) |
| Russian | ryan | 😠 anger | *Как ты смеешь так со мной разговаривать? Это неприемлемо!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ru_ryan_anger.wav) |
| Japanese | ono_anna | 😢 sad | *私が持っていたものを全て失って、もうどうすればいいのか分からない。* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ja_ono_anna_sad.wav) |
| Japanese | ono_anna | 😨 fear | *家に誰かいる、足音が聞こえた……怖くてどうすればいいのか分からない。* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ja_ono_anna_fear.wav) |
| Korean | sohee | 😠 anger | *네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ko_sohee_anger.wav) |

- **Emotions:** 6 primaries — `sad · joy · anger · fear · disgust · surprise` (synonyms like `happy`/`angry`
  work too) — **plus 7 blended "dyads"** (below).
- **The recipe:** a **preset voice** → pure **STEER** (the steering vector @ w12, clean in every language); a
  **cloned voice** → **COMBINE** (the language `.expr` + steer). Use the **native preset per language** (JA
  `ono_anna`, KO `sohee`, ZH `vivian`, EN/Romance `ryan`); the engine prints a hint. Full recipe →
  [docs/emotion-THE-recipe.md](docs/emotion-THE-recipe.md).
- **Works in every Qwen3-TTS language** (EN, IT, DE, ZH, RU, KO, JA, ES, FR, PT) — just set `-l <Language>`.

#### Blended emotions (dyads) · new

Emotion steering **directions add**: summing two primary vectors yields a coherent *new* emotion. Seven
ear-validated **Plutchik dyads** ship as first-class `--emotion` values — no new capture, no fine-tune:

| Dyad | = blend of | Reads as | Listen (English, ryan) |
|---|---|---|---|
| `contempt`    | anger + disgust    | sneering disdain | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_contempt.wav) |
| `awe`         | fear + surprise    | hushed wonder | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_awe.wav) |
| `nostalgia`   | joy + sad          | bittersweet fondness | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_nostalgia.wav) |
| `disapproval` | surprise + sad     | let-down reproach | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_disapproval.wav) |
| `remorse`     | sad + disgust      | guilty regret | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_remorse.wav) |
| `outrage`     | anger + surprise   | indignant shock | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_outrage.wav) |
| `despair`     | fear + sad         | hopeless dread | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/dyad_despair.wav) |

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English --emotion contempt \
    --text "Oh, sure, that's a truly brilliant idea." -o contempt.wav
```

#### Inline emotion switching — many emotions from ONE prompt · new

Write `[emotion]` tags **inside `--text`** and the engine switches emotion **sentence by sentence in a single
generation** — any primary or dyad, no flags. Clean at the seams, one output file:

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English -T 1.1 --text \
  "[contempt] Oh, sure, that's a brilliant idea. [nostalgia] We used to spend every summer by the sea. [despair] And now there's nothing left." \
  -o switch.wav
```

**🔊 Hear the switch happen inside one prompt:**

| Prompt (inline `[tags]`) | Listen |
|---|---|
| `[contempt]` Oh, sure, that's a brilliant idea. `[nostalgia]` We used to spend every summer by the sea. `[despair]` And now there's nothing left. | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/switch_en_contempt-nostalgia-despair.wav) |
| `[sad]` I really thought this would work out. `[disgust]` But the whole thing is rotten. `[contempt]` As if they ever cared. | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/switch_en_sad-disgust-contempt.wav) |
| *(Italian)* `[outrage]` Hanno annullato tutto senza dirci niente. `[remorse]` Continuo a pensare a cosa ho detto. `[awe]` Poi ho alzato lo sguardo e sono rimasto senza parole. | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_dyads/switch_it_outrage-remorse-awe.wav) |

> Inline `[emotion]` uses the same steering recipe as `--emotion`, applied per sentence. `[neutral]` resets to
> no emotion. Combine with a global `--emotion` and paralinguistic `[laugh]`/`[sigh]` tags freely.
- **Paralinguistics → inline `[tags]`, also automatic · 🧪 Alpha.** Write `[laugh]`, `[sigh]`, `[yawn]`, `[wow]`, `[giggle]` or `[scoff]` in
  `--text` and the engine performs the event (it picks the onomatopoeia anchor + the right seed per voice for
  you) — no flags. `[wow]`/`[yawn]`/`[scoff]` compose well with the matching `--emotion`; `[giggle]` is best
  **standalone** (stacking it with `--emotion joy` over-drives the laugh). *Alpha quality:* hit-or-miss across
  voices/languages (laughs land best); expect misses for now:
  ```bash
  ./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
      --text "Che giornata... [sigh] non ce la faccio più. [laugh]" -o para.wav
  ```

#### Emotion + paralinguistics together (experimental 🧪)

You can put a paralinguistic `[tag]` **inside an emotional sentence** and get both at once — e.g. `--emotion joy`
+ `[laugh]`. When a `[tag]` is present the engine switches the emotion to its **COMBINE** stack (the `.expr`
language-correction keeps the event from drifting the accent) and rides the laugh/sigh steering vector at the
per-voice weight (ryan w6, others w8). The pure-emotion path (no tag) is unchanged. This is **still a bit
unstable** across some languages/voices (work in progress) — the clearest results are `[laugh]`/`[sigh]` on
`ryan`/`vivian`. Reproduce with `make emotion-para-demo`.

| Language | Voice | Emotion + tag | Text | Listen |
|----------|-------|---------------|------|--------|
| Italian | ryan (preset) | 😄 joy + `[laugh]` | *Non ci posso credere, `[laugh]` è la notizia più bella della mia vita!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_joy_laugh.wav) |
| Italian | ryan (preset) | 😢 sad + `[sigh]` | *Ho perso tutto quello che avevo, `[sigh]` e adesso non so più cosa fare.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_it_sad_sigh.wav) |
| English | ryan (preset) | 😄 joy + `[laugh]` | *I can't believe it, `[laugh]` this is the best news of my whole life!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/ryan_en_joy_laugh.wav) |
| French | vivian | 😢 sad + `[sigh]` | *J'ai tout perdu, `[sigh]` et maintenant je ne sais plus quoi faire.* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/fr_vivian_sad_sigh.wav) |
| Spanish | vivian | 😄 joy + `[laugh]` | *No me lo puedo creer, `[laugh]` ¡es la mejor noticia de mi vida!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/es_vivian_joy_laugh.wav) |
| Italian | galatea (cloned voice) | 😄 joy + `[laugh]` | *Non ci posso credere, `[laugh]` è la notizia più bella della mia vita!* | [▶ play](https://github.com/gabriele-mastrapasqua/qwen3-tts/raw/main/samples/emotion_examples/galatea_it_joy_laugh.wav) |

#### Tuning the `--instruct` (strength & speed)

`--instruct` is an **optional** vivid English (or Chinese) line that rides on top of the emotion recipe (1.7B; it
matters for **cloned voices** and preset+instruct — preset pure-emotion needs none). Two things it controls well:

- **Strength** — a stronger, more vivid instruct pushes emotion harder. Escalate the wording when you want more
  (e.g. anger gets raspier and angrier); back off if it starts to sound noisy. Write it as **plain prose**, not a
  parameter list.
- **Speed** — say it in words: `"… and speak a little faster"` / `"speak slowly"` shifts pacing (~±15 %);
  `"in a higher voice"` lifts the pitch a touch.

```bash
# mild vs strong wording — same recipe, more push
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English -T 1.1 --emotion anger \
    --instruct "Speak in an absolutely furious, explosive, screaming rage, voice cracking with violent anger." \
    --text "How dare you talk to me like that? I will not accept this!" -o anger.wav
```

> **Don't** use a slot/parameter template (`VoiceStyle: … Tempo: +15% Pitch: higher`). Qwen3-TTS does **not** parse
> the slots — `Tempo:+40%` even comes out *slower*. Plain vivid prose wins. Full findings + a ready per-emotion
> `strong`/`very-strong` instruct library → [docs/emotion-instruct-control.md](docs/emotion-instruct-control.md).

**Assets:** `bash download_assets.sh` (introduced in the setup callout above) fetches the `.expr` packs;
`--verify` re-checks integrity. Full set ≈ 1.4 GB; Italian-only emotion needs just `italian_csp_topk6.expr` (203 MB).

<details>
<summary><b>Under the hood</b> — what each file is, and the manual override flags (advanced)</summary>

You normally never touch these — `--emotion` and `[tags]` load the right ones. But if you want to tune by hand:

| File | Where | What it is | Manual flag |
|------|-------|------------|-------------|
| `.expr` | `presets/expr/` (HF) | **Per-language emotion fine-tune** — a weight-delta on the Talker's emotion layers; fixes/renders the language *and* gives the base emotion. | `--expr <file> --expr-weight <m>` |
| `.qlsteer` | `presets/steer/emotion/` (git) | **Emotion steering vector** — an inference-time activation direction (per voice × emotion). Changes no weights; carries the emotion, transfers cross-voice/language. | `--ml-steer <file> --ml-range 21-25 --ml-weight <w>` |
| `.qlsteer` | `presets/steer/paraling/` (git) | **Paralinguistic vector** — `laugh_vs_cry`, `sigh_vs_laugh`. Speaker/language-agnostic. | (auto via `[laugh]`/`[sigh]`) |
| `.qamp` | `presets/steer/paraling/` (git) | **Raw activation fingerprint** — the source a `.qlsteer` is built from (reproducibility). | (build input) |

A manual `--expr` / `--ml-steer` always **overrides** the `--emotion` auto-router. Validated recipe (2026-06-29):
**preset → pure STEER `ryan_<emo>` @ `w12`** (clean, every language; w10 also good); **clone → COMBINE** (language
`.expr` + steer). Use the native preset per language (JA `ono_anna`, KO `sohee`, ZH `vivian`, EN/Romance `ryan`).
**Train your own `.expr` for any language** with [`training/expressivity-lora/`](training/expressivity-lora/).
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

# With emotion (same recipe as the CLI --emotion; joy/sad/angry/calm/…)
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"What a wonderful day!","speaker":"ryan","language":"English","emotion":"joy"}' -o joy.wav

# Stream with real-time playback (emotion works on the streaming path too)
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you?","emotion":"sad"}' | \
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

### 🖥️ GPU backends — Apple Metal & NVIDIA CUDA (opt-in)

Optional `--backend metal|cuda` runs the **whole fused pipeline resident on the GPU** (weights + KV +
activations on device, one command buffer / step). The CPU path stays the default — GPU is purely additive.
Full numbers: [Metal / Apple Silicon](docs/hardware-testing.md) · [CUDA / NVIDIA](docs/cuda-performance.md).

**Apple Metal** — `make metal CC=clang`, then `QWEN_METAL_FUSED_TALKER=1 ./qwen_tts --backend metal`.
**Single-stream latency** (one request — CLI, or a warm `--serve` server; the two match):

| Device | 0.6B RTF | 1.7B RTF | Streaming TTFA (single client) |
|---|---|---|---|
| **Apple M1** 8-core (dev box) | ~0.60 (int4) | — | **469 ms** (0.6B) |
| **Apple M2 Pro** 16-core GPU | **0.36–0.39** | **0.48–0.53** | **314 ms** / 517 ms |

RTF = processing_time ÷ audio_duration (**< 1.0 = faster than real time**); TTFA = time to first audio for a
single `--stream` client, **warm server** (the first request after startup pays a one-time weight→GPU-buffer
upload — e.g. ~3.6 s cold vs 469 ms warm on M1-0.6B). Metal beats the native M2 CPU path ~1.5–2×; **int8 is the sweet spot** on Apple Silicon
(bandwidth-rich → int4's nibble-unpack doesn't pay). Resident decode is bit-identical to the CPU path.
*(Multi-user concurrency → the batching table below.)*

**NVIDIA CUDA** — `make cuda` (resident fused + cuBLAS pointwise convs + CUDA graphs), 1.7B, on a mainstream
**~270 GB/s GPU (RTX 4060-class)**. **Single-stream latency** (one request):

| Config | RTF (single stream) |
|---|---|
| Resident fused (`--quant-mixed`: int4 Talker + int8 CP) | **0.44** |

Decode is bandwidth-bound, so RTF scales with memory bandwidth: RTX 3060 ~0.33 · 4070 ~0.24 · 4090 ~0.12
(estimates; 4060-class is measured).

**Throughput — server request-batching** (`--serve --batch-size N`, continuous batching + per-request
streaming). Batching is a **throughput / parallelism** lever, *not* a per-request speedup — it serves N
concurrent users in roughly the time of one by reading each weight once for all B sequences (matvec → matmat).
**CPU, CUDA and Metal all batch:**

| Backend | Batch speedup | Notes |
|---|---|---|
| **CUDA** (RTX 4060-class) | **~3.35× at B=8** | per-step (Talker 4.1× · CP 2.7×), ~3× end-to-end; output bit-identical solo-vs-batch |
| **Apple Metal** (M2 Pro) | **~2.8× at B=4** | 0.6B 2.81× · 1.7B 2.82× (consistent); batch output bit-identical to single-stream |
| **CPU** | ~N on bandwidth-bound x86 servers | ~1× on cache-rich M1 (single-stream is already fast) |

## Documentation

| Guide | Contents |
|-------|----------|
| [Voice Cloning](docs/voice-cloning.md) | Reference audio tips, ECAPA-TDNN internals, model comparison, samples |
| [Custom Voices](docs/custom-voices.md) | `.qvoice` format, delta vs standard, managing profiles, troubleshooting |
| [HTTP Server](docs/server.md) | All endpoints, request body, streaming, server performance |
| [Server request-batching](docs/server-batching.md) | vLLM-style `--batch-size N`: serve N concurrent users together, continuous batching, per-request streaming |
| [VoiceDesign](docs/voice-design.md) | Creating voices from text descriptions |
| [Emotion — THE recipe](docs/emotion-THE-recipe.md) | The one-and-only `--emotion` recipe: preset → STEER @ w12, clone → COMBINE; native preset per language. Single source of truth |
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
