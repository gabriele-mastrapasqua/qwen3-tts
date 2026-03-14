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
make test-large-int8  # Run 1.7B INT8 quantization tests (Italian + English, seed 42)
make test-large-int4  # Run 1.7B INT4 quantization tests (Italian + English, seed 42)
make test-large-quant # Run all 1.7B quantization tests (INT8 + INT4)
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
  --temperature <f>          Sampling temperature (default: 0.5)
  --top-k <n>                Top-k sampling (default: 50)
  --top-p <f>                Top-p nucleus sampling (default: 1.0)
  --rep-penalty <f>          Repetition penalty (default: 1.05)
  --max-tokens <n>           Max audio tokens (default: 8192)
  --max-duration <secs>      Max audio duration in seconds
  --seed <n>                 Random seed for reproducible output
  --voice-design             VoiceDesign mode (create voice from --instruct)
  --ref-audio <path>         Reference audio for voice cloning (Base model)
  --xvector-only             Use speaker embedding only (no ref text/codes)
  --save-voice <path>        Save voice profile (.qvoice with ICL data, or .bin for raw embedding)
  --load-voice <path>        Load voice profile (.qvoice or legacy .bin)
  --max-ref-duration <secs>  Max ref audio for embedding (default: 15, 0=all)
  -j, --threads <n>          Worker threads (default: 4)
  --stream                   Stream audio (decode chunks during generation)
  --stdout                   Output raw s16le PCM to stdout (implies --stream)
  --stream-chunk <n>         Frames per stream chunk (default: 10 = 0.8s)
  --int8                     INT8 quantized Talker + Code Predictor (1.7B recommended)
  --int4                     Q4_0 quantized Talker (1.7B only, experimental)
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

### Weight Quantization (1.7B model)

The `--int8` and `--int4` flags quantize Talker weights at load time, reducing memory usage and (for INT8) improving speed on the 1.7B model. These flags have no meaningful effect on the 0.6B model (matrices too small to be bandwidth-bound).

```bash
# INT8 — recommended for 1.7B (15% Talker speedup, good quality)
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int8 -o hello.wav

# INT4 — experimental, smaller memory but no speed gain
./qwen_tts -d qwen3-tts-1.7b --text "Hello world" --int4 -o hello.wav
```

**Comparison (1.7B, Italian, seed=42, Apple M1 16 GB, 4 threads):**

| Config | Talker ms/f | Total time | RTF | Talker RAM |
|--------|-------------|------------|-----|------------|
| BF16 (default) | ~80 ms/f | ~13s | ~4.3 | 2.8 GB (mmap) |
| **INT8 (recommended)** | **~67 ms/f** | **~11s** | **~3.6** | **1.4 GB** |
| INT4 (experimental) | ~83 ms/f | ~14s | ~4.5 | 0.7 GB |

> **Recommendation:** Use `--int8` for the 1.7B model. It gives 15% Talker speedup with
> good audio quality. INT4 saves memory but is slightly *slower* due to nibble unpacking
> overhead. For maximum speed, use the 0.6B model (RTF ~1.3–1.7 vs 3.6 for 1.7B INT8).
>
> On systems with 16+ GB free RAM, expected performance is better than shown above
> (our test machine had high system memory pressure from other applications).
> Projected RTF with free RAM: **0.6B ~1.3, 1.7B BF16 ~3.0, 1.7B INT8 ~2.5**.

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
> process (temperature=0.5 by default). Some seeds sound better than others. If a particular
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

Clone any voice from a short reference audio clip. **Requires a Base model** — the
CustomVoice models (0.6B, 1.7B) do NOT support voice cloning.

**The 1.7B-Base model produces significantly better voice clones** than the 0.6B-Base.
The 1.7B has a richer speaker embedding (2048-dim vs 1024-dim) and a larger transformer
that better preserves the original speaker's timbre, pitch, and speaking style. The
0.6B-Base works well for preset voices but loses some fidelity when cloning unseen voices.
Use 1.7B-Base when voice clone quality matters most.

```bash
# Download the Base model (has speaker encoder for voice cloning)
./download_model.sh --model base-small   # 0.6B-Base (faster, good quality)
./download_model.sh --model base-large   # 1.7B-Base (slower, best clone quality)

# Clone a voice from a WAV file (1.7B recommended for best results)
./qwen_tts -d qwen3-tts-1.7b-base --text "Hello, this is my cloned voice." \
    --ref-audio reference.wav -o cloned.wav

# Clone with Italian text
./qwen_tts -d qwen3-tts-1.7b-base --text "Ciao, questa e la mia voce clonata." \
    --ref-audio reference.wav -o cloned_it.wav
```

#### Reusable Voice Profiles (`.qvoice`)

Save a cloned voice to a `.qvoice` file once, then reuse it for any text without
re-processing the reference audio. This skips mel spectrogram extraction, speaker
encoding, and speech encoding — giving a **2x speedup** on subsequent generations.

```bash
# Step 1: Create a .qvoice profile from reference audio (no --text needed)
#   Encodes audio and saves: speaker embedding + ICL codec tokens + transcript
#   Use the same Base model you'll generate with (0.6B or 1.7B)
./qwen_tts -d qwen3-tts-1.7b-base \
    --ref-audio reference.wav --ref-text "Exact transcript of the reference audio." \
    --save-voice my_voice.qvoice

# Step 2: Reuse the saved voice for any new text (no ref audio needed)
./qwen_tts -d qwen3-tts-1.7b-base \
    --load-voice my_voice.qvoice \
    --text "A completely different sentence." -o output.wav
```

**Performance comparison** (Apple M1 8-core, 4 threads, 0.6B-Base, ~4s output):

| Mode | Prefill | Total | RTF | Notes |
|------|---------|-------|-----|-------|
| From WAV (`--ref-audio`) | 2.8s | 21.6s | 4.91 | Mel + speaker enc + speech enc + generate |
| From `.qvoice` (`--load-voice`) | 1.7s | 9.8s | 2.23 | Load file + generate (no audio processing) |

The `.qvoice` format (v2) is compact (~20-50KB for a typical 3-10s reference clip) and
stores the speaker embedding dimension (`enc_dim`) in the header.

> **Important:** `.qvoice` files are **model-specific** — a file created with 0.6B-Base
> (`enc_dim=1024`) cannot be used with 1.7B-Base (`enc_dim=2048`) and vice versa. The tool
> will show a clear error if there is a mismatch. Re-create the `.qvoice` with the matching
> Base model. Legacy v1 `.qvoice` files (without `enc_dim` header) are still supported.

#### Managing Voice Profiles

```bash
# List all .qvoice files in a directory
./qwen_tts --list-voices ./my_voices/

# Inspect a single .qvoice file
./qwen_tts --list-voices my_voice.qvoice

# Delete a voice profile
./qwen_tts --delete-voice ./my_voices/old_voice.qvoice
```

These commands don't require a model — they read/manage the `.qvoice` files directly.

#### Legacy format

Raw speaker embedding files (`.bin`) are still supported for backward compatibility,
but they only store the x-vector (lower quality than ICL mode with `.qvoice`):

```bash
./qwen_tts -d qwen3-tts-0.6b-base --text "Hello" \
    --ref-audio reference.wav --save-voice my_voice.bin -o out.wav
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
> a voice embedding from the reference audio.

#### How Voice Cloning Works

The speaker encoder is an **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and
Aggregation in TDNN) network that converts reference audio into a fixed-size speaker embedding:

1. **Mel spectrogram**: The reference audio is converted to a 128-band mel spectrogram at 24 kHz
   (hop size 256 = ~94 frames/second).
2. **TDNN + SE-Res2Net blocks**: Four convolutional blocks extract speaker-characteristic features
   across time — capturing pitch, timbre, and speaking style.
3. **Attentive Statistics Pooling**: Computes a weighted mean and standard deviation over the
   **entire temporal sequence**. This is the key step: longer audio means the pooling sees more
   variation in the speaker's voice (different intonations, pitch ranges, speaking styles),
   producing a richer and more representative embedding.
4. **FC projection**: The pooled statistics (3072-dim) are projected to the final embedding
   dimension (1024 for 0.6B, 2048 for 1.7B) and injected into the transformer prompt.

#### Reference Audio Duration

More reference audio generally produces better voice clones. The attentive pooling layer benefits
from seeing diverse speech patterns — monotone input yields a flatter embedding, while varied
speech with different intonations captures the speaker's full vocal range.

| Duration | Mel frames | Quality | Notes |
|----------|-----------|---------|-------|
| 5-10s | 470-940 | Good | Minimum for recognizable clone |
| 15-20s | 1400-1880 | Better | Covers basic vocal range |
| **30s** | **2810** | **Recommended** | Good balance of quality and speed |
| 45s+ | 4200+ | Best | Diminishing returns, slower extraction |

By default, the first **30 seconds** of reference audio are used. Use `--max-ref-duration 0`
to process the entire file, or set a custom limit (e.g., `--max-ref-duration 45`).

**Tips for best results:**
- **Use clean audio without background music or noise.** The speaker encoder processes the
  raw mel spectrogram and cannot separate voice from background — music, ambient noise, or
  other speakers will be captured as part of the speaker embedding and reproduced as artifacts
  in the output. If your reference has background noise, pre-process it with a voice separation
  tool (e.g., [demucs](https://github.com/facebookresearch/demucs)) before cloning.
- Include varied speech (questions, statements, different emotions) rather than monotone reading
- 24 kHz WAV is ideal; other sample rates will be rejected (convert with `ffmpeg -i input.wav -ar 24000 output.wav`)

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

#### Model Comparison for Voice Cloning

| | 0.6B-Base | 1.7B-Base |
|---|---|---|
| **Speaker embedding dim** | 1024 | 2048 |
| **Transformer hidden** | 1024 | 2048 |
| **Clone fidelity** | Good | Best |
| **Speed (Apple M1)** | RTF ~1.5–1.7 | RTF ~3.2–4.1 |
| **Best for** | Fast cloning, acceptable quality | Maximum voice fidelity |
| **Style control (`--instruct`)** | Not supported | Supported |

The 1.7B-Base produces noticeably more faithful voice clones — the 2048-dim speaker
embedding captures twice the vocal detail (timbre, pitch contour, breathiness,
speaking rhythm) compared to the 0.6B's 1024-dim embedding. The larger transformer
also has more capacity to condition its output on these speaker characteristics.

For the technical details of how the speaker encoder works, see the
[voice cloning internals blog post](blog/voice-cloning-internals.md).

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
| **First call** | 12.2s → RTF 1.50 | 20.0s → RTF 1.28 |
| **Warm call** | 11.3s → RTF 1.39 | 19.7s → **RTF 1.26** |

The first request pays a one-time cost for tokenizer parsing (~200ms) and warming the
OS page cache for mmap'd weights. Warm calls benefit from cached tokenizer, resident
weight pages, pre-allocated buffers, **LRU text embedding cache** (~8MB for 2048 tokens),
and **decoder thread overlap** (speech decoder runs in background during generation).

### RTF across modes (0.6B, Apple M1 8-core 16 GB, 4 threads)

|  | Short text (~5–8s audio) | Long text (~16s audio) |
|---|---|---|
| **CLI** | RTF 1.4–1.7 | ~RTF 1.3 |
| **CLI `--stream`** | RTF 1.4–1.7 | ~RTF 1.3 |
| **Server (cold)** | RTF 1.50 | RTF 1.28 |
| **Server (warm)** | RTF 1.39 | **RTF 1.26** |

Streaming mode has **identical performance** to normal mode — the speech decoder
runs in a pipeline thread in both cases. Longer audio amortizes fixed costs (prefill).
Server mode adds warm caches, embedding cache, and decoder thread overlap on top.

#### Full request body

```json
{
  "text": "Hello world",
  "speaker": "ryan",
  "language": "English",
  "instruct": "Speak cheerfully",
  "seed": 42,
  "temperature": 0.5,
  "top_k": 50,
  "top_p": 1.0,
  "rep_penalty": 1.05
}
```

All fields except `text` are optional. Defaults: speaker=ryan, language=English,
temperature=0.5, top_k=50, top_p=1.0, rep_penalty=1.05, seed=random.
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

- **0.6B**: RTF ~1.3–1.7 depending on audio length and mode (server warm + long text = best)
- **1.7B**: RTF ~3.0–4.3 (BF16), ~2.5–3.6 with `--int8` (recommended)
- Bottleneck is the Code Predictor (15 sequential autoregressive passes per frame)
- SIMD-optimized kernels (NEON on ARM, AVX on x86) for BF16/INT8 matrix-vector operations
- Optional INT8/INT4 Talker quantization for the 1.7B model (see [Weight Quantization](#weight-quantization-17b-model))
- Cache-line aligned buffers (64B `posix_memalign`) for optimal BLAS/SIMD throughput
- Multi-threaded inference via GCD (`dispatch_apply`) on macOS, pthreads on Linux

### Per-component breakdown (0.6B, seed 42, CLI single run)

| Component | Short text (4.7s audio) | Long text (16.8s audio) |
|-----------|------------------------|------------------------|
| Prefill | 1,647ms | ~1,040ms |
| Talker | 24.6 ms/frame | ~22 ms/frame |
| Code Predictor | 76.3 ms/frame | ~60 ms/frame |
| Speech Decoder | overlapped (512ms drain) | overlapped |
| **Total** | **8.2s → RTF 1.74** | **~20s → ~RTF 1.4** |

The speech decoder runs in a **background thread** during generation, overlapping
most of its work with Talker+CP. Only the final "drain" (waiting for the last chunk)
adds to wall time. Prefill and per-frame costs amortize over longer audio, with
an asymptotic RTF approaching ~1.0.

### Optimization history

Starting from a baseline of **RTF ~3.5** (CLI), the following optimizations brought
performance to **RTF ~1.3–1.7** (up to 2.7x total speedup):

| Optimization | Speedup | Technique |
|---|---|---|
| Cache-line alignment (`posix_memalign(64)`) | **24%** | Aligned all BLAS/SIMD buffers and KV caches |
| Decoder thread overlap | **14-19%** | Speech decoder runs in background thread during generation |
| SIMD speech decoder | **11%** | Replaced scalar RMSNorm, RoPE, attention with NEON/AVX |
| Persistent prefill buffers | **38% server** | Reuse buffers across generations (zero malloc in decode) |
| Text embedding cache | **14% server** | LRU cache for token embeddings (skip 2 matvec per cached token) |
| Batched VQ projection | minor | BLAS sgemm instead of per-frame scalar matvec |
| Pre-allocated sampling buffers | minor | Zero per-token malloc in generation loop |
| Top-k quickselect | **4× sampling** | O(n) quickselect replaces O(kn) selection sort |
| Streaming pipeline parallelism | **RTF 2.0→1.4** | Decoder thread runs in streaming mode too |

All optimizations are cross-platform (POSIX standard `posix_memalign`, conditional NEON/AVX).
See [blog/optimization-notes.md](blog/optimization-notes.md) for the full story.

### How does CPU compare to GPU?

For context, here's how the official Python + PyTorch implementation performs on GPUs:

| Hardware | 0.6B RTF (short) | 0.6B RTF (long) | Notes |
|----------|------------------|-----------------|-------|
| **This project (C, Apple M1 CPU)** | **1.39** | **1.26** | **Pure C, server warm, no GPU** |
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
> an Apple M1 (8-core, 2020, LPDDR4X) gets RTF 1.3–1.7 — on older, slower hardware.
> That's the difference between optimized C with SIMD (NEON/AVX) + BLAS and Python with PyTorch overhead.
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
