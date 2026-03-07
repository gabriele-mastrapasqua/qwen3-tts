# Qwen3-TTS Pure C Implementation

This is a C implementation of the inference pipeline for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (both 0.6B and 1.7B). It has zero external dependencies beyond the C standard library and a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux). Audio is generated frame by frame as the model runs.

## Audio Samples

All samples generated with the 0.6B model on Apple M1 at ~0.7x realtime:

| Language | Speaker | Sample | Text |
|----------|---------|--------|------|
| English | ryan | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/english_ryan.wav) | *Hello, this is a test of the text to speech system.* |
| Italian | vivian | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/italian_vivian.wav) | *Buongiorno a tutti, questa è una dimostrazione del sistema di sintesi vocale.* |
| Spanish | ryan | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/spanish_ryan.wav) | *Hola, esta es una demostración del sistema de síntesis de voz.* |
| Portuguese | ryan | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/portuguese_ryan.wav) | *Olá, esta é uma demonstração do sistema de síntese de voz.* |
| French | ryan | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/french_ryan.wav) | *Bonjour à tous, ceci est une démonstration du système de synthèse vocale.* |
| German | ryan | [▶ listen](https://github.com/gabriele-mastrapasqua/qwen3-tts/releases/download/v0.1-samples/german_ryan.wav) | *Guten Tag, dies ist eine Demonstration des Sprachsynthesesystems.* |

> Or clone and play locally: `afplay samples/english_ryan.wav` (macOS) / `aplay samples/english_ryan.wav` (Linux)

**Important**: this implementation explicitly **avoids implementing support for MPS**. Following the same philosophy as [qwen-asr](https://github.com/antirez/qwen-asr): TTS systems are important pieces of infrastructure often run on remote Linux servers. Adding the MPS target would focus efforts too much on Apple hardware, so for now it is skipped. The code runs well on Apple hardware anyway (NEON optimized). MPS support may be added later when other optimizations are mature.

## Quick Start

```bash
# Build
make blas

# Download a model (interactive selector: small=0.6B, large=1.7B)
./download_model.sh

# Synthesize speech
./qwen_tts -d qwen3-tts-0.6b --text "Hello, how are you today?" -o hello.wav

# Play the output (macOS)
afplay hello.wav

# Play the output (Linux)
aplay hello.wav
```

## Features

- **Almost zero dependencies**: Pure C implementation. Only needs BLAS (Accelerate on macOS, OpenBLAS on Linux).
- **Both models**: Automatically detects 0.6B or 1.7B from the weight files.
- **CustomVoice speakers**: 9 preset voices selectable by name (`-s ryan`, `-s vivian`, etc.).
- **Multilingual**: 10 languages supported, selectable with `-l English`, `-l Italian`, etc.
- **Sampling control**: Temperature, top-k, top-p, and repetition penalty are configurable.
- **Memory-mapped weights**: BF16 weights are mmap'd directly from safetensors files — loading is near-instant.
- **WAV output**: 24 kHz, 16-bit PCM, mono.

## Usage

```bash
./qwen_tts [options]

Options:
  -d, --model-dir <path>     Model directory (required)
  --text <string>            Text to synthesize (required)
  -o, --output <path>        Output WAV file (default: output.wav)
  -s, --speaker <name>       Speaker name (ryan, vivian, serena, aiden, etc.)
  -l, --language <lang>      Target language (English, Italian, Chinese, etc.)
  --temperature <f>          Sampling temperature (default: 0.9)
  --top-k <n>                Top-k sampling (default: 50)
  --top-p <f>                Top-p nucleus sampling (default: 1.0)
  --rep-penalty <f>          Repetition penalty (default: 1.05)
  --max-tokens <n>           Max audio tokens to generate (default: 8192)
  --silent                   Suppress status output on stderr
  --debug                    Verbose internal diagnostics
```

### Examples

```bash
# Basic English synthesis
./qwen_tts -d qwen3-tts-0.6b --text "The quick brown fox jumps over the lazy dog." -o fox.wav

# Choose a speaker by name (-s) and language (-l)
./qwen_tts -d qwen3-tts-0.6b -s ryan -l English \
    --text "Hello, this is a test of the text to speech system." -o test_en.wav

# Italian with a specific speaker
./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian \
    --text "Ciao, questa è una prova del sistema di sintesi vocale." -o test_it.wav

# Switch voice: same text, different speaker (female)
./qwen_tts -d qwen3-tts-0.6b -s vivian -l Italian \
    --text "Buongiorno, come state oggi? Spero tutto bene." -o test_it_vivian.wav

# Lower temperature for more deterministic output
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --temperature 0.7 -o hello.wav
```

Available speakers: `serena`, `vivian`, `uncle_fu`, `ryan`, `aiden`, `ono_anna`, `sohee`, `eric`, `dylan`.

You can also use `make test-en`, `make test-it-ryan`, `make test-it-vivian`, or `make test-all` to quickly run pre-configured tests.

## Building

```bash
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux)
make debug      # Debug build with AddressSanitizer
make clean      # Clean build artifacts
make info       # Show build configuration
```

For Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

## Model Architecture

Qwen3-TTS is a text-to-speech model available in 0.6B and 1.7B parameter variants:

**Pipeline:**
```
Text → BPE Tokenizer → Talker (LLM) → Code Predictor (MTP) → Speech Decoder (ConvNet) → 24 kHz WAV
```

| Component | Architecture |
|-----------|-------------|
| Talker | 28-layer Qwen3 with GQA, per-head Q/K RMSNorm, NeoX RoPE, SwiGLU |
| Code Predictor | 5-layer transformer, 15 sequential passes per audio frame |
| Speech Decoder | Causal ConvNet, 16 codebook RVQ, 480x upsampling to 24 kHz |

| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Talker layers | 28 | 28 |
| Talker dim | 1024 | 2048 |
| Code Predictor layers | 5 | 5 |
| Codebooks | 16 × 2048 entries | 16 × 2048 entries |
| Frame rate | 12.5 Hz | 12.5 Hz |
| Output sample rate | 24 kHz | 24 kHz |
| Weight format | BF16 | BF16 |
| Languages | 10 | 10 |

Supported languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

## Memory Requirements

- **0.6B**: ~3 GB total (model weights + runtime buffers)
- **1.7B**: ~8 GB total (model weights + runtime buffers)

Safetensors are memory-mapped. Large weights (Talker, Code Predictor) remain as BF16 mmapped.
Speech decoder weights are loaded to F32.

## Credits & Acknowledgments

This project builds on the work of several teams and individuals:

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team at Alibaba — the model architecture, weights, and research behind the text-to-speech system. Models available on [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice).
  - *Qwen3-TTS Technical Report* — [arXiv:2505.08reduced](https://arxiv.org/abs/2505.15894)
- **[qwen-asr](https://github.com/antirez/qwen-asr)** by [antirez](https://github.com/antirez) — a pure C implementation of Qwen2-Audio ASR that directly inspired this project's architecture: mmap'd safetensors, BF16 NEON kernels, threading via `dispatch_apply`, and the overall approach of writing minimal C inference engines. Much of the safetensors loader and kernel scaffolding is derived from qwen-asr.
- **[Qwen2.5](https://github.com/QwenLM/Qwen2.5)** by the Qwen team — the base LLM architecture (GQA, RoPE, SwiGLU) used in the Talker and Code Predictor components.

## License

MIT
