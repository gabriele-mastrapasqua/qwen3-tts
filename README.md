# Qwen3-TTS Pure C Implementation

[![Build](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/build.yml)
[![CodeQL](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/codeql.yml)
[![Memory Safety](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml/badge.svg)](https://github.com/gabriele-mastrapasqua/qwen3-tts/actions/workflows/safety.yml)

A lightweight, cross-platform C inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (0.6B and 1.7B). No Python, no PyTorch, no ONNX runtime — just C, a BLAS library, and raw model weights.

The engine runs the complete TTS pipeline: BPE tokenization, a 28-layer causal transformer (Talker), a multi-pass code predictor, and a convolutional speech decoder. Weights are memory-mapped directly from safetensors files in BF16, so loading is near-instant and memory usage stays low.

## Audio Samples

All samples generated with the 0.6B model (RTF ~1.3–2.0, Apple M1):

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

## Features

- **Pure C, minimal dependencies** — Only requires a C compiler and BLAS (Accelerate on macOS, OpenBLAS on Linux). No Python runtime needed.
- **Cross-platform** — Runs on macOS (ARM/x86) and Linux (ARM/x86). NEON and AVX SIMD paths included. See [Windows (WSL2)](#windows-wsl2--beta) for Windows support (beta).
- **Both model sizes** — Automatically detects 0.6B or 1.7B from weight files.
- **9 preset voices** — Selectable by name: `ryan`, `vivian`, `serena`, `aiden`, `eric`, `dylan`, `uncle_fu`, `ono_anna`, `sohee`.
- **10 languages** — English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.
- **Memory-mapped weights** — BF16 safetensors are mmap'd directly. The 0.6B model needs ~3 GB, the 1.7B needs ~8 GB.
- **Configurable sampling** — Temperature, top-k, top-p, and repetition penalty.
- **24 kHz WAV output** — 16-bit PCM, mono.

## Building

### macOS

```bash
make blas    # Uses Accelerate framework (ships with Xcode)
```

### Linux

```bash
# Install OpenBLAS
sudo apt install libopenblas-dev    # Ubuntu/Debian
sudo dnf install openblas-devel     # Fedora/RHEL

make blas
```

### Windows (WSL2) — Beta

WSL2 runs a real Linux kernel, so the build is identical to native Linux.
Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu if you haven't already.

> **Beta:** These instructions have not been tested on a Windows machine yet.
> The codebase builds and runs on native Linux, so WSL2 should work out of the box.
> Please open an issue if you run into problems.

```bash
# In a WSL2 terminal (Ubuntu)
sudo apt update && sudo apt install build-essential libopenblas-dev

git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts
make blas

# Download a model and generate speech
./download_model.sh --model small
./qwen_tts -d qwen3-tts-0.6b --text "Hello from Windows!" -o hello.wav
```

Playing audio from WSL2:

```bash
# Option 1: Open with Windows media player (works out of the box)
powershell.exe Start-Process "$(wslpath -w hello.wav)"

# Option 2: Use aplay if PulseAudio/PipeWire is configured in WSL2
aplay hello.wav

# Option 3: Copy to Windows and play manually
cp hello.wav /mnt/c/Users/$USER/Desktop/
```

### Other build targets

```bash
make debug      # Debug build with AddressSanitizer
make clean      # Clean build artifacts
make info       # Show build configuration
```

### Testing

```bash
make test-small       # Run 0.6B tests (English, Italian, multiple speakers)
make test-large       # Run 1.7B tests (config check, English, Italian, instruct styles)
make test-regression  # Cross-model regression checks (safetensors, config parsing)
make test-all         # Run everything (0.6B + 1.7B + regression)
make test-serve          # HTTP server integration test (health, speakers, TTS)
make test-serve-bench    # Server benchmark: 2 runs, same seed, verify bit-identical output
make test-serve-openai   # OpenAI-compatible /v1/audio/speech endpoint test
make test-serve-parallel # 2 concurrent requests, verify both complete
make test-serve-all      # Run all server tests
make serve               # Start HTTP server on port 8080
```

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
  --temperature <f>          Sampling temperature (default: 0.9)
  --top-k <n>                Top-k sampling (default: 50)
  --top-p <f>                Top-p nucleus sampling (default: 1.0)
  --rep-penalty <f>          Repetition penalty (default: 1.05)
  --max-tokens <n>           Max audio tokens (default: 8192)
  --max-duration <secs>      Max audio duration in seconds
  --seed <n>                 Random seed for reproducible output
  --voice-design             VoiceDesign mode (create voice from --instruct)
  --ref-audio <path>         Reference audio for voice cloning (Base model)
  --xvector-only             Use speaker embedding only (no ref text/codes)
  --save-voice <path>        Save speaker embedding to file for reuse
  --load-voice <path>        Load speaker embedding (skip extraction)
  --max-ref-duration <secs>  Max ref audio for embedding (default: 15, 0=all)
  -j, --threads <n>          Worker threads (default: 4)
  --stream                   Stream audio (decode chunks during generation)
  --stdout                   Output raw s16le PCM to stdout (implies --stream)
  --stream-chunk <n>         Frames per stream chunk (default: 10 = 0.8s)
  --silent                   Suppress status output
  --debug                    Verbose diagnostics
```

### Examples

```bash
# Basic English
./qwen_tts -d qwen3-tts-0.6b --text "The quick brown fox jumps over the lazy dog." -o fox.wav

# Italian with a male voice
./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian \
    --text "Ciao, questa e una prova del sistema di sintesi vocale." -o test_it.wav

# French with a female voice
./qwen_tts -d qwen3-tts-0.6b -s vivian -l French \
    --text "Bonjour, comment allez-vous aujourd'hui?" -o test_fr.wav

# Lower temperature for more deterministic output
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --temperature 0.7 -o hello.wav

# Use the larger model for higher quality
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" -o hello_large.wav

# Style/emotion control with --instruct (1.7B model only)
./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
    --text "I cannot believe you did that to me." \
    --instruct "Speak in a very angry and aggressive tone" -o angry.wav

./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
    --text "I cannot believe you did that to me." \
    --instruct "Speak very slowly and softly, in a sad whisper" -o whisper.wav

./qwen_tts -d qwen3-tts-1.7b -s ryan -l English \
    --text "I cannot believe you did that to me." \
    --instruct "Speak in a very happy, cheerful and excited tone" -o happy.wav
```

> **Note:** The `--instruct` flag only works with the 1.7B model. The 0.6B model does not
> support style control and will ignore the instruction.

### Seed & Reproducibility

By default, each run uses a time-based random seed, so the same text produces slightly different audio each time. Use `--seed` for reproducible output:

```bash
# Same seed → same audio every time
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --seed 42 -o hello.wav

# Different seeds → different prosody, pacing, and intonation
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --seed 1 -o v1.wav
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --seed 2 -o v2.wav
```

> **Note:** Audio quality varies across seeds — this is inherent to the model's sampling
> process (temperature=0.9 by default). Some seeds sound better than others. If a particular
> generation sounds off, try a different seed or lower the temperature. Duration also varies
> significantly (3-7x range for the same text), which is normal model behavior.

### VoiceDesign

Create entirely new voices from natural language descriptions using the VoiceDesign model:

```bash
# Download the VoiceDesign model
./download_model.sh --model voice-design

# Deep British male
./qwen_tts -d qwen3-tts-voice-design -l English \
    --instruct "A deep male voice with a British accent, speaking slowly and calmly" \
    --text "Hello, this is a test of the voice design system." -o british.wav

# Young energetic female
./qwen_tts -d qwen3-tts-voice-design -l English \
    --instruct "Young energetic female, cheerful and fast-paced" \
    --text "Oh my gosh, this is so exciting!" -o cheerful.wav

# Chinese loli voice
./qwen_tts -d qwen3-tts-voice-design -l Chinese \
    --instruct "萝莉女声，撒娇稚嫩" \
    --text "你好，这是一个语音设计的测试。" -o loli.wav
```

> **Note:** VoiceDesign requires a 1.7B model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`).
> It does **not** work with the 0.6B model — the engine will refuse to run if you try.
> The model type is auto-detected from the config. `--instruct` is required
> to describe the desired voice. No `--speaker` is needed.

### Voice Cloning

Clone any voice from a short reference audio clip using the Base model:

```bash
# Download the Base model (has speaker encoder for voice cloning)
./download_model.sh --model base-small

# Clone a voice from a WAV file
./qwen_tts -d qwen3-tts-0.6b-base --text "Hello, this is my cloned voice." \
    --ref-audio reference.wav -o cloned.wav

# Clone with Italian text
./qwen_tts -d qwen3-tts-0.6b-base --text "Ciao, questa e la mia voce clonata." \
    --ref-audio reference.wav -o cloned_it.wav

# Save voice embedding for reuse (avoids re-extracting each time)
./qwen_tts -d qwen3-tts-0.6b-base --text "Hello" \
    --ref-audio reference.wav --save-voice my_voice.bin -o out.wav

# Load saved voice (faster — skips mel spectrogram + speaker encoder)
./qwen_tts -d qwen3-tts-0.6b-base --text "Another sentence" \
    --load-voice my_voice.bin -o out2.wav
```

#### Quick Demo

```bash
# Clone from a sample WAV (outputs to samples/)
make demo-clone

# Clone from your own audio
make demo-clone REF=my_voice.wav

# Custom text too
make demo-clone REF=my_voice.wav TEXT="Hello from my cloned voice!"
```

#### Voice Clone Samples

| Input | Cloned Output | Text |
|-------|---------------|------|
| [reference (movie clip)](samples/10s_back_down_the_road.wav) | [english](samples/clone_output_en.wav) | *I love programming in C, it gives you complete control over the machine.* |
| [reference (movie clip)](samples/10s_back_down_the_road.wav) | [italian](samples/clone_output_it.wav) | *Buongiorno, questa e una dimostrazione della clonazione vocale.* |

> **Note:** Voice cloning requires a **Base** model (`Qwen3-TTS-12Hz-0.6B-Base` or `1.7B-Base`),
> not the CustomVoice model. The Base model includes an ECAPA-TDNN speaker encoder that extracts
> a voice embedding from the reference audio. A few seconds of clear speech is sufficient.
>
> By default, only the first **15 seconds** of reference audio are used for the speaker embedding.
> This is enough for high-quality cloning and keeps extraction fast. Use `--max-ref-duration 0`
> to process the entire file, or set a custom limit (e.g., `--max-ref-duration 30`).

#### Reference Audio Format

The reference audio **must be 24 kHz WAV** (PCM, mono or stereo, 16-bit or 32-bit).
If your audio is in a different format (MP3, Opus, OGG, or a different sample rate),
convert it first with ffmpeg:

```bash
# Convert any audio file to 24 kHz mono WAV
ffmpeg -i input.mp3 -ar 24000 -ac 1 output.wav
ffmpeg -i input.opus -ar 24000 -ac 1 output.wav

# Even WAV files may need resampling (e.g., 16 kHz → 24 kHz)
ffmpeg -i voice_16k.wav -ar 24000 output.wav
```

This is required because the ECAPA-TDNN speaker encoder uses a mel spectrogram
computed at 24 kHz with specific FFT parameters (n_fft=1024, hop=256, 128 mels).
A mismatched sample rate would produce incorrect mel features and a bad voice embedding.

### Streaming

```bash
# Stream to WAV file (audio written progressively during generation)
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stream -o hello.wav

# Pipe raw PCM to audio player for real-time playback
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stdout | \
    play -t raw -r 24000 -e signed -b 16 -c 1 -   # macOS (requires sox)

# Linux real-time playback
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stdout | \
    aplay -f S16_LE -r 24000 -c 1

# Adjust chunk size (larger = fewer decodes, smaller = lower latency)
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --stream --stream-chunk 5 -o hello.wav
```

> **Note:** Streaming decodes audio progressively every N frames (default: 10 = 0.8s).
> First audio is available within ~1 second. `--stdout` outputs raw signed 16-bit
> little-endian mono PCM at 24 kHz — pipe it to any audio player.

### HTTP Server

The server loads the model once at startup and keeps weights in memory across requests.
The tokenizer is also cached after the first call, so **subsequent requests skip all loading
overhead and go straight to inference** (~5-6s per short sentence on 0.6B, 4 threads, Apple M1 8-core 16 GB).

```bash
# Start server (model loaded once, shared across requests)
./qwen_tts -d qwen3-tts-0.6b --serve 8080
```

#### Generate speech (full WAV)

```bash
# Minimal — defaults to speaker=ryan, language=English
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"Hello, how are you today?"}' -o output.wav

# With explicit options
curl -s http://localhost:8080/v1/tts \
  -d '{"text":"Ciao, come stai?","speaker":"vivian","language":"Italian"}' \
  -o ciao.wav

# OpenAI-compatible endpoint (drop-in replacement)
curl -s http://localhost:8080/v1/audio/speech \
  -d '{"input":"Hello world","voice":"ryan"}' -o output.wav
```

#### Streaming playback

The `/v1/tts/stream` endpoint returns chunked raw PCM (s16le, 24 kHz, mono) as it generates.
First audio arrives within ~1 second; pipe it directly to an audio player for real-time playback:

```bash
# macOS — real-time playback via ffplay
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  ffplay -f s16le -ar 24000 -ac 1 -nodisp -autoexit -

# macOS — real-time playback via sox
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  play -t raw -r 24000 -e signed -b 16 -c 1 -

# Linux — real-time playback via aplay
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello, how are you today?"}' | \
  aplay -f S16_LE -r 24000 -c 1

# Save raw PCM to file
curl -sN http://localhost:8080/v1/tts/stream \
  -d '{"text":"Hello"}' -o output.raw
# Convert to WAV: ffmpeg -f s16le -ar 24000 -ac 1 -i output.raw output.wav
```

#### Other endpoints

```bash
# List speakers
curl -s http://localhost:8080/v1/speakers | python3 -m json.tool

# Health check
curl -s http://localhost:8080/v1/health
```

#### Server performance (0.6B, Apple M1 8-core 16 GB, 4 threads)

Same text, same seed (`--seed 42`), identical output (bit-for-bit):

| | Short text (~8s audio) | Long text (~16s audio) |
|---|---|---|
| **First call** | 15.1s → RTF 1.85 | 20.9s → RTF 1.33 |
| **Warm call** | 11.5s → RTF 1.41 | 20.6s → **RTF 1.31** |

The first request pays a one-time cost for tokenizer parsing (~200ms) and warming the
OS page cache for mmap'd weights. Warm calls benefit from cached tokenizer, resident
weight pages, pre-allocated buffers, and the **LRU text embedding cache** (token
embeddings are computed once and reused across requests, avoiding redundant bf16 matvec
projections — ~8MB RAM for 2048 cached tokens).

### RTF across modes (0.6B, Apple M1 8-core 16 GB, 4 threads)

|  | Short text (~5–8s audio) | Long text (~16s audio) |
|---|---|---|
| **CLI** | RTF 2.01 | RTF 1.42 |
| **Server (cold)** | RTF 1.85 | RTF 1.33 |
| **Server (warm)** | RTF 1.41 | **RTF 1.31** |

Longer audio amortizes fixed costs (prefill, speech decoder). Server mode adds
warm caches and embedding cache on top.

#### Full request body

```json
{
  "text": "Hello world",
  "speaker": "ryan",
  "language": "English",
  "instruct": "Speak cheerfully",
  "seed": 42,
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 1.0,
  "rep_penalty": 1.05
}
```

All fields except `text` are optional. Defaults: speaker=ryan, language=English,
temperature=0.9, top_k=50, top_p=1.0, rep_penalty=1.05, seed=random.
Each request starts from clean defaults — parameters do not leak between requests.

## How It Works

```
Text --> BPE Tokenizer --> Talker (LLM) --> Code Predictor --> Speech Decoder --> 24 kHz WAV
```

| Component | What it does |
|-----------|-------------|
| **Talker** | 28-layer Qwen3 transformer with GQA, RoPE, SwiGLU. Generates one audio frame token per step. |
| **Code Predictor** | 5-layer transformer running 15 sequential passes per frame. Predicts the remaining 15 codebook entries for each frame. |
| **Speech Decoder** | Causal ConvNet with 16-codebook RVQ dequantization and 480x upsampling. Converts discrete codes to raw audio waveform. |

| | 0.6B | 1.7B |
|-----------|------|------|
| **Talker** | | |
| Hidden dim | 1024 | 2048 |
| Heads (Q/KV) | 16/8 | 16/8 |
| Intermediate | 3072 | 6144 |
| Layers | 28 | 28 |
| **Code Predictor** | | |
| Hidden dim | 1024 | 1024 |
| Heads | 16 | 16 |
| Layers | 5 | 5 |
| MTP projection | — | 2048→1024 |
| **General** | | |
| Parameters | ~600M | ~1.7B |
| Memory usage | ~3 GB | ~8 GB |
| Weight format | BF16 | BF16 |

The Code Predictor has the same architecture in both models (hidden=1024, 5 layers). On the 1.7B, a linear projection bridges the larger Talker hidden dim (2048) down to the CP's 1024. All dimensions are auto-detected from weight files — no recompilation needed to switch models.

## Performance

Benchmarked on Apple M1 8-core, 16 GB RAM, 4 threads:

- **0.6B**: RTF ~1.3–2.0 depending on audio length and mode (server warm + long text = best)
- Bottleneck is the Code Predictor (15 sequential autoregressive passes per frame)
- SIMD-optimized kernels (NEON on ARM, AVX on x86) for BF16 matrix-vector operations
- Cache-line aligned buffers (64B `posix_memalign`) for optimal BLAS/SIMD throughput
- Multi-threaded inference via GCD (`dispatch_apply`) on macOS, pthreads on Linux

### Per-component breakdown (0.6B, seed 42, CLI single run)

| Component | Short text (4.7s audio) | Long text (16.8s audio) |
|-----------|------------------------|------------------------|
| Prefill | 1,633ms | 1,040ms |
| Talker | 21.0 ms/frame | 22.0 ms/frame |
| Code Predictor | 69.3 ms/frame | 60.1 ms/frame |
| Speech Decoder | 2,359ms | 5,313ms |
| **Total** | **9.5s → RTF 2.01** | **23.9s → RTF 1.42** |

Prefill and speech decoder are fixed costs that amortize over longer audio.
Per-frame decode (Talker + CP) is ~82 ms/frame, which sets the asymptotic RTF at ~1.0
for sufficiently long generations.

### RTF across modes (0.6B, Apple M1 8-core 16 GB, 4 threads)

|  | Short text (~5–8s audio) | Long text (~16s audio) |
|---|---|---|
| **CLI** | RTF 2.01 | RTF 1.42 |
| **Server (cold)** | RTF 1.77 | RTF 1.55 |
| **Server (warm)** | RTF 1.39 | **RTF 1.34** |

Server warm calls benefit from cached tokenizer, resident mmap'd weight pages,
and pre-allocated prefill/sampling buffers. Longer audio amortizes fixed costs
(prefill, speech decoder) over more frames.

### Optimization history

Starting from a baseline of **RTF ~3.5** (CLI), the following optimizations brought
performance to **RTF ~1.3–2.0** (up to 2.5x total speedup):

| Optimization | Speedup | Technique |
|---|---|---|
| Cache-line alignment (`posix_memalign(64)`) | **24%** | Aligned all BLAS/SIMD buffers and KV caches |
| NEON speech decoder | **11%** | Replaced scalar RMSNorm, RoPE, attention with NEON |
| Persistent prefill buffers | **38% server** | Reuse buffers across generations (zero malloc in decode) |
| Text embedding cache | **14% server** | LRU cache for token embeddings (skip 2 matvec per cached token) |
| Batched VQ projection | minor | BLAS sgemm instead of per-frame scalar matvec |
| Pre-allocated sampling buffers | minor | Zero per-token malloc in generation loop |

All optimizations are cross-platform (POSIX standard `posix_memalign`, conditional NEON/AVX).
See [blog/optimization-notes.md](blog/optimization-notes.md) for the full story.

### How does CPU compare to GPU?

For context, here's how the official Python + PyTorch implementation performs on GPUs:

| Hardware | 0.6B RTF (short) | 0.6B RTF (long) | Notes |
|----------|------------------|-----------------|-------|
| **This project (C, Apple M1 CPU)** | **1.41** | **1.31** | **Pure C, server warm, no GPU** |
| Python + PyTorch (Ryzen 9 7950X CPU) | 4.5–5.8 | — | Official Python, CPU-only, no GPU |
| NVIDIA RTX 3090 | 0.52 | 0.68 | Python + PyTorch + FlashAttention 2 |
| NVIDIA RTX 4090 | 0.38 | 0.45 | Python + PyTorch + FlashAttention 2 |
| NVIDIA A100 | 0.28 | 0.35 | Data center GPU |
| NVIDIA H100 | 0.22 | 0.28 | Data center GPU |

> RTF = Real-Time Factor = processing_time / audio_duration. Lower is faster; <1.0 means faster than real-time.
> GPU benchmarks from [qwen3-tts.app](https://qwen3-tts.app/blog/qwen3-tts-performance-benchmarks-hardware-guide-2026).
>
> Two things stand out:
>
> **We're 3–4x faster than Python on CPU.** The official Python + PyTorch implementation
> on a Ryzen 9 7950X (16-core Zen 4, 2022, DDR5) gets RTF 4.5–5.8. Our pure C engine on
> an Apple M1 (8-core, 2020, LPDDR4X) gets RTF 1.3–2.0 — on older, slower hardware.
> That's the difference between optimized C with NEON/BLAS and Python with PyTorch overhead.
>
> **GPUs get worse on long text, we get better.** GPU RTF degrades 18–31% from short
> to long text (attention scales quadratically even with FlashAttention). Our RTF *improves*
> 7% on long text because fixed costs (prefill, speech decoder) amortize over more frames
> while our per-token decode is constant-time (linear matvec, no quadratic attention).
>
> For a CPU-only engine on a 2020 laptop, being within 2x of a consumer GPU (RTX 3090)
> and 3–4x faster than the official Python CPU path is a solid result.

## Credits & Acknowledgments

- **Salvatore Sanfilippo ([antirez](https://github.com/antirez))** — This project wouldn't exist without his [qwen-asr](https://github.com/antirez/qwen-asr), a pure C Qwen2-Audio ASR engine that proved you can do real neural inference in plain C with mmap'd safetensors, BF16 NEON kernels, and zero dependencies. The entire architecture of this TTS engine — the approach, the style, the philosophy of minimal C inference — is directly inspired by his work. If you like this project, go star qwen-asr first.
- **Michael Abrash** — His *[Graphics Programming Black Book](https://www.jagregory.com/abrash-black-book/)* (1997) shaped how we think about performance. The chapters on data alignment, struct layout, and cache-friendly access patterns for the 386/486 are still relevant today — we got a **24% speedup** from cache-line alignment (`posix_memalign(64)`), applying the same principles Abrash taught 30 years ago to modern SIMD and BLAS. The specific rules changed (64-byte cache lines instead of dword alignment), but the instinct is the same.
- **John Carmack** — His `.plan` files and QuakeCon talks on micro-optimization and cache friendliness were a constant reference. Where Abrash gave you the systematic rules and benchmarks, Carmack showed you the mindset: always think about how data flows through the CPU.
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team at Alibaba — the model architecture, weights, and research. Models on [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). [Paper](https://arxiv.org/abs/2505.15894).
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team — the base LLM architecture (GQA, RoPE, SwiGLU) used in the Talker and Code Predictor.

## License

MIT
