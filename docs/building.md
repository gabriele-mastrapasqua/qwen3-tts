# Building

## Dependencies

Only a C compiler and BLAS are required:
- **macOS**: Accelerate framework (ships with Xcode)
- **Linux**: OpenBLAS

LZ4 compression (for `.qvoice` voice files) is embedded in the repo — no separate install needed.

## macOS

```bash
make blas    # Uses Accelerate framework
```

## Linux

```bash
# Install OpenBLAS
sudo apt install libopenblas-dev    # Ubuntu/Debian
sudo dnf install openblas-devel     # Fedora/RHEL

make blas
```

## Windows (WSL2) — Beta

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

### Playing audio from WSL2

```bash
# Option 1: Open with Windows media player (works out of the box)
powershell.exe Start-Process "$(wslpath -w hello.wav)"

# Option 2: Use aplay if PulseAudio/PipeWire is configured in WSL2
aplay hello.wav

# Option 3: Copy to Windows and play manually
cp hello.wav /mnt/c/Users/$USER/Desktop/
```

## Other Build Targets

```bash
make debug      # Debug build with AddressSanitizer
make clean      # Clean build artifacts
make info       # Show build configuration
```

## Testing

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
make bench               # RTF benchmark (short+long, normal+stream, both models)
make bench-full          # Full benchmark (+ server, instruct, INT8, .qvoice)
```
