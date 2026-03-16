# Qwen3-TTS Pure C Implementation

[![Build](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml)
[![CodeQL](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml)
[![Memory Safety](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml)

A lightweight, cross-platform C inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (0.6B and 1.7B). No Python, no PyTorch, no ONNX runtime — just C, a BLAS library, and raw model weights.

The engine runs the complete TTS pipeline: BPE tokenization, a 28-layer causal transformer (Talker), a multi-pass code predictor, and a convolutional speech decoder. Weights are memory-mapped directly from safetensors files in BF16, so loading is near-instant and memory usage stays low.

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
- **Cross-platform** — macOS (ARM/x86) and Linux (ARM/x86). NEON and AVX SIMD paths. [Windows/WSL2](docs/building.md) beta.
- **Both model sizes** — Automatically detects 0.6B or 1.7B from weight files.
- **9 preset voices** — `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.
- **10 languages** — English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Memory-mapped weights** — BF16 safetensors mmap'd directly. 0.6B ~3 GB, 1.7B ~8 GB.
- **Voice cloning** — Clone any voice from a short WAV clip. Save as reusable `.qvoice` profile. [Full docs](docs/voice-cloning.md)
- **Custom voices with Delta `.qvoice`** — Bit-identical cloned voices on CustomVoice model. Create once, use forever — with style control, streaming, server. [Full docs](docs/custom-voices.md)
- **Voice management** — List, inspect, delete `.qvoice` profiles (`--list-voices`, `--delete-voice`). No model required.
- **Style control** — `--instruct` for emotion/style on 1.7B: angry, whisper, cheerful, and more.
- **VoiceDesign** — Create new voices from text descriptions. [Full docs](docs/voice-design.md)
- **HTTP server** — `/v1/tts`, `/v1/tts/stream`, OpenAI-compatible `/v1/audio/speech`. [Full docs](docs/server.md)
- **Streaming** — Real-time audio via `--stream` (WAV) or `--stdout` (raw PCM).
- **INT8/INT4 quantization** — 15% speedup on 1.7B with `--int8`. [Full docs](docs/quantization.md)
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
  --save-voice <path>        Save voice profile (.qvoice)
  --load-voice <path>        Load voice profile (.qvoice)
  --target-cv <dir>          CV model dir for delta encoding (bit-identical cross-model)
  --list-voices <dir>        List .qvoice files in directory (no model needed)
  --delete-voice <path>      Delete a .qvoice file
  --voice-name <name>        Name for the voice (stored in .qvoice metadata)
  --voice-design             VoiceDesign mode (create voice from --instruct)
  --stream                   Stream audio (decode chunks during generation)
  --stdout                   Output raw s16le PCM to stdout (implies --stream)
  --int8                     INT8 quantized (1.7B recommended)
  --int4                     Q4_0 quantized (1.7B only, experimental)
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

> Full guide: delta vs standard, format internals, troubleshooting → [docs/custom-voices.md](docs/custom-voices.md)

### HTTP Server

```bash
# Start server
./qwen_tts -d qwen3-tts-0.6b --serve 8080

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
| Hidden dim | 1024 | 2048 |
| Heads (Q/KV) | 16/8 | 16/8 |
| Layers | 28 | 28 |
| Code Predictor | 1024 hidden, 5 layers | 1024 hidden, 5 layers (+2048→1024 projection) |
| Memory | ~3 GB | ~8 GB |

## Performance

Benchmarked on Apple M1 8-core, 16 GB RAM, 4 threads:

| Mode | 0.6B RTF | 1.7B BF16 RTF | 1.7B INT8 RTF |
|------|----------|---------------|---------------|
| **CLI** | 1.4–1.7 | ~4.3 | ~3.6 |
| **CLI `--stream`** | 1.4–1.7 | ~4.3 | ~3.6 |
| **Server (cold)** | 1.50 | — | 3.57 |
| **Server (warm)** | **1.39** | — | 3.32 |
| **Server stream** | 1.48 | — | 3.18 |

RTF = processing_time / audio_duration. Lower is better; <1.0 = faster than real-time.

Longer audio improves RTF (fixed costs amortize): 0.6B server warm on long text reaches **RTF 1.26**.
Streaming has identical performance to normal mode. `--int8` gives 15% Talker speedup on 1.7B
([details](docs/quantization.md)).

**vs other implementations:**

| Hardware | 0.6B RTF | Notes |
|----------|----------|-------|
| **This project (C, Apple M1 CPU)** | **1.26–1.39** | Pure C, no GPU |
| Python + PyTorch (Ryzen 9 7950X CPU) | 4.5–5.8 | Official Python, CPU-only |
| NVIDIA RTX 3090 | 0.52–0.68 | Python + PyTorch + FlashAttention 2 |

3–4x faster than Python on CPU. Within 2x of an RTX 3090 — on a 2020 laptop with no GPU.

> Per-component breakdown, full GPU table, optimization history → [docs/performance.md](docs/performance.md)

## Documentation

| Guide | Contents |
|-------|----------|
| [Voice Cloning](docs/voice-cloning.md) | Reference audio tips, ECAPA-TDNN internals, model comparison, samples |
| [Custom Voices](docs/custom-voices.md) | `.qvoice` format, delta vs standard, managing profiles, troubleshooting |
| [HTTP Server](docs/server.md) | All endpoints, request body, streaming, server performance |
| [VoiceDesign](docs/voice-design.md) | Creating voices from text descriptions |
| [Quantization](docs/quantization.md) | INT8/INT4, comparison table, recommendations |
| [Performance](docs/performance.md) | RTF benchmarks, component breakdown, CPU vs GPU, optimization history |
| [Building](docs/building.md) | All platforms, build targets, testing |

### Blog Posts

| Post | Topic |
|------|-------|
| [Voice Cloning Internals](blog/voice-cloning-internals.md) | ECAPA-TDNN architecture deep-dive |
| [Cross-Model Voice Analysis](blog/cross-model-voice-analysis.md) | Why delta format works (weight analysis) |
| [Optimization Notes](blog/optimization-notes.md) | RTF 3.5 → 1.3: the full optimization story |

## Credits & Acknowledgments

- **Salvatore Sanfilippo ([antirez](https://github.com/antirez))** — This project wouldn't exist without his [qwen-asr](https://github.com/antirez/qwen-asr), a pure C Qwen2-Audio ASR engine that proved you can do real neural inference in plain C with mmap'd safetensors, BF16 NEON kernels, and zero dependencies. The entire architecture of this TTS engine — the approach, the style, the philosophy of minimal C inference — is directly inspired by his work. If you like this project, go star qwen-asr first.
- **Michael Abrash** — His *[Graphics Programming Black Book](https://www.jagregory.com/abrash-black-book/)* (1997) shaped how we think about performance. The chapters on data alignment, struct layout, and cache-friendly access patterns for the 386/486 are still relevant today — we got a **24% speedup** from cache-line alignment (`posix_memalign(64)`), applying the same principles Abrash taught 30 years ago to modern SIMD and BLAS.
- **John Carmack** — His `.plan` files and QuakeCon talks on micro-optimization and cache friendliness were a constant reference. Where Abrash gave you the systematic rules and benchmarks, Carmack showed you the mindset: always think about how data flows through the CPU.
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team at Alibaba — the model architecture, weights, and research. Models on [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). [Paper](https://arxiv.org/abs/2505.15894).
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team — the base LLM architecture (GQA, RoPE, SwiGLU) used in the Talker and Code Predictor.

## License

MIT
