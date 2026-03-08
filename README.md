# Qwen3-TTS Pure C Implementation

A lightweight, cross-platform C inference engine for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (0.6B and 1.7B). No Python, no PyTorch, no ONNX runtime — just C, a BLAS library, and raw model weights.

The engine runs the complete TTS pipeline: BPE tokenization, a 28-layer causal transformer (Talker), a multi-pass code predictor, and a convolutional speech decoder. Weights are memory-mapped directly from safetensors files in BF16, so loading is near-instant and memory usage stays low.

## Audio Samples

All samples generated with the 0.6B model at ~0.7x realtime:

| Language | Speaker | Sample | Text |
|----------|---------|--------|------|
| English | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/english_ryan.wav) | *Hello, this is a test of the text to speech system.* |
| Italian | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/italian_ryan.wav) | *Buongiorno a tutti, questa e una dimostrazione del sistema di sintesi vocale.* |
| Italian | vivian | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/italian_vivian.wav) | *Buongiorno a tutti, questa e una dimostrazione del sistema di sintesi vocale.* |
| Spanish | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/spanish_ryan.wav) | *Hola, esta es una demostracion del sistema de sintesis de voz.* |
| Portuguese | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/portuguese_ryan.wav) | *Ola, esta e uma demonstracao do sistema de sintese de voz.* |
| French | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/french_ryan.wav) | *Bonjour a tous, ceci est une demonstration du systeme de synthese vocale.* |
| German | ryan | [listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/german_ryan.wav) | *Guten Tag, dies ist eine Demonstration des Sprachsynthesesystems.* |

> Clone and play locally: `afplay samples/english_ryan.wav` (macOS) or `aplay samples/english_ryan.wav` (Linux)

## Quick Start

```bash
# Clone and build
git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts
make blas

# Download a model (interactive: small=0.6B, large=1.7B, voice-design)
./download_model.sh

# Synthesize speech
./qwen_tts -d qwen3-tts-0.6b --text "Hello, how are you today?" -o hello.wav
```

## Features

- **Pure C, minimal dependencies** — Only requires a C compiler and BLAS (Accelerate on macOS, OpenBLAS on Linux). No Python runtime needed.
- **Cross-platform** — Runs on macOS (ARM/x86) and Linux (ARM/x86). NEON and AVX SIMD paths included. See [Windows (WSL2)](#windows-wsl2) for Windows support.
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

### Windows (WSL2)

```bash
# Open a WSL2 terminal (Ubuntu recommended)
sudo apt update && sudo apt install build-essential libopenblas-dev
git clone https://github.com/gabriele-mastrapasqua/qwen3-tts.git
cd qwen3-tts
make blas
./download_model.sh --model small
./qwen_tts -d qwen3-tts-0.6b --text "Hello from Windows!" -o hello.wav
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
make test-serve       # HTTP server integration test
make serve            # Start HTTP server on port 8080
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

> **Note:** VoiceDesign requires the `Qwen3-TTS-12Hz-1.7B-VoiceDesign` model.
> The model type is auto-detected from the config. `--instruct` is required
> to describe the desired voice. No `--speaker` is needed.

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

```bash
# Start server (model loaded once, shared across requests)
./qwen_tts -d qwen3-tts-0.6b --serve 8080

# Generate speech (returns WAV)
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","speaker":"ryan","language":"English"}' \
  -o output.wav

# Streaming (returns chunked raw PCM as it generates)
curl -X POST http://localhost:8080/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","speaker":"ryan"}' \
  -o output.raw

# OpenAI-compatible endpoint (drop-in replacement)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"ryan"}' \
  -o output.wav

# List speakers
curl http://localhost:8080/v1/speakers

# Health check
curl http://localhost:8080/v1/health
```

The full request body for `/v1/tts`:
```json
{
  "text": "Hello world",
  "speaker": "ryan",
  "language": "English",
  "instruct": "Speak cheerfully",
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 1.0,
  "rep_penalty": 1.05
}
```

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
| Talker hidden dim | 1024 | 2048 |
| Talker heads (Q/KV) | 16/8 | 32/8 |
| Parameters | ~600M | ~1.7B |
| Memory usage | ~3 GB | ~8 GB |
| Weight format | BF16 | BF16 |

All model dimensions are read from the weight files at load time — no recompilation needed to switch between 0.6B and 1.7B.

## Performance

Benchmarked with 4 threads on CPU:

- **0.6B**: ~0.7x realtime (generates 1 second of audio in ~1.4 seconds)
- Bottleneck is the Code Predictor (15 sequential autoregressive passes per frame)
- SIMD-optimized kernels (NEON on ARM, AVX on x86) for BF16 matrix-vector operations
- Multi-threaded inference via GCD (`dispatch_apply`) on macOS, pthreads on Linux

## Credits & Acknowledgments

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team at Alibaba — the model architecture, weights, and research. Models on [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice). [Paper](https://arxiv.org/abs/2505.15894).
- **[qwen-asr](https://github.com/antirez/qwen-asr)** by [antirez](https://github.com/antirez) — a pure C Qwen2-Audio ASR engine that directly inspired this project's architecture: mmap'd safetensors, BF16 NEON kernels, and the overall approach of writing minimal C inference engines.
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team — the base LLM architecture (GQA, RoPE, SwiGLU) used in the Talker and Code Predictor.

## License

MIT
