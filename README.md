# Qwen3-TTS Pure C Implementation

This is a C implementation of the inference pipeline for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models (both 0.6B and 1.7B). It has zero external dependencies beyond the C standard library and a BLAS implementation (Accelerate on macOS, OpenBLAS on Linux). Audio is generated frame by frame as the model runs.

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
- **CustomVoice speakers**: 9 preset speakers selectable via `--speaker 0..8`.
- **Language control**: `--language English` sets the target language.
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
  --speaker <id>             Speaker ID 0-8 (default: 0, CustomVoice only)
  --language <lang>          Target language (default: auto from text)
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
# Basic synthesis
./qwen_tts -d qwen3-tts-0.6b --text "The quick brown fox jumps over the lazy dog." -o fox.wav

# Different speaker
./qwen_tts -d qwen3-tts-1.7b --text "Good morning!" --speaker 3 -o morning.wav

# Italian with specific language
./qwen_tts -d qwen3-tts-1.7b --text "Buongiorno, come stai?" --language Italian -o ciao.wav

# Lower temperature for more deterministic output
./qwen_tts -d qwen3-tts-0.6b --text "Hello world" --temperature 0.7 -o hello.wav
```

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

## License

MIT
