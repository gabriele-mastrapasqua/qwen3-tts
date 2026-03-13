This file is the practical guide for agents working on this repository.
It is intentionally implementation-oriented: what to change, where, how to test,
and which behaviors are considered contractually stable.

## Project Scope

Pure C inference engine for Qwen3-TTS text-to-speech models:
- `Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`

Primary target is CPU inference (BLAS + architecture-specific SIMD paths).

## Current Status

- **All components verified correct** (Talker, Code Predictor, Speech Decoder — bit-identical to Python)
- **Performance**: RTF ~1.3–1.7 on Apple M1 8-core, 16 GB RAM (4 threads)
- **Dev hardware**: Apple M1 8-core, 16 GB RAM — all benchmarks reference this machine

## Active Branches

- **`main`**: Stable, production-ready code
- **`feat/labs`**: Experimental branch for architecture tweaks, performance experiments, mixed optimizations, and general R&D. Use this for testing before merging to main.

## CI/CD (planned, not yet implemented)

After implementation, GitHub Actions will provide:
- **Build matrix**: Linux x86/ARM, macOS ARM/x86, Windows/WSL2 (manual trigger only)
- **CodeQL + clang-tidy**: Auto on PR to main (buffer overflow, UB, security checks)
- **ASan/UBSan**: Auto on PR to main (runtime memory safety with real model)
- **Benchmarks**: Per-runner CPU dump + ms/f breakdown as JSON artifacts
- **Releases**: Static binaries per platform on tag push (`v*`)
- See PLAN.md Phase 9 for full details

## Source Of Truth

When docs and code disagree, trust these files first:
- CLI behavior and options: `main.c`
- Public API and runtime state: `qwen_tts.h`
- Pipeline orchestration + prompt construction: `qwen_tts.c`
- Talker (LLM) forward pass + KV cache: `qwen_tts_talker.c`
- Code Predictor (MTP) forward pass: `qwen_tts_code_predictor.c`
- Speech tokenizer decoder (ConvNet): `qwen_tts_speech_decoder.c`
- Sampling strategies: `qwen_tts_sampling.c`
- WAV output: `qwen_tts_audio.c`
- Kernel dispatch and hot loops: `qwen_tts_kernels*.c`, `qwen_tts_kernels_impl.h`
- Build targets: `Makefile`

Architecture/background references:
- `MODEL.md`

## User-Facing Behavior Contract (Do Not Break)

- Default output is a WAV file at 24 kHz, 16-bit PCM, mono.
- `--silent` must still write the WAV output file.
- `--silent` suppresses status/debug noise (stderr), not the audio output.
- Without `--debug`, stderr should be concise:
  - model loading info
  - final inference summary lines
- `--debug` enables verbose internal diagnostics.
- `--speaker` selects a preset speaker ID (0-8 for CustomVoice models).

## Model + Inference Facts

- Model variant is auto-detected from weights (0.6B vs 1.7B).
- Talker uses causal Qwen3 with KV cache, GQA, SwiGLU, RoPE.
- Code Predictor runs 15 sequential passes per audio frame.
- Speech decoder is a causal ConvNet with 480x upsampling.
- Large weights are bf16 mmapped and consumed via bf16 kernels.
- Output audio is 24 kHz, generated from 16-codebook discrete tokens at 12.5 Hz.

## Important Defaults

From `qwen_tts_load()` and CLI:
- Speaker ID: `0` (first preset voice)
- Temperature: `0.5`
- Top-k: `50`
- Top-p: `1.0`
- Repetition penalty: `1.05`
- Max new tokens: `8192`
- Code Predictor temperature: `0.0` (greedy)
- Code Predictor top-k: `50`
- Output file: `output.wav`

## Repository Map

- `main.c`
  - CLI parsing, defaults, reporting
- `qwen_tts.c`
  - high-level synthesis pipeline
  - prompt construction (ChatML format)
  - generation loop (Talker + Code Predictor + Speech Decoder)
- `qwen_tts_talker.c`
  - Talker LLM load + prefill + token step + KV cache
- `qwen_tts_code_predictor.c`
  - Code Predictor (MTP) load + forward (15 passes per frame)
- `qwen_tts_speech_decoder.c`
  - speech tokenizer decoder load + forward
  - codebook embedding lookup + RVQ sum
  - pre-transformer layers
  - ConvNet upsampling blocks
- `qwen_tts_audio.c`
  - WAV writer (24 kHz, 16-bit PCM, mono)
- `qwen_tts_sampling.c`
  - temperature, top-k, top-p, repetition penalty
- `qwen_tts_tokenizer.c`
  - tokenizer encode (text to token IDs)
- `qwen_tts_safetensors.c`
  - safetensors loading and mmap
- `qwen_tts_kernels.c`
  - common math, threading, BLAS paths
- `qwen_tts_kernels_generic.c`
  - generic hot kernels
- `qwen_tts_kernels_neon.c`
  - ARM NEON hot kernels
- `qwen_tts_kernels_avx.c`
  - x86 AVX hot kernels
- `qwen_tts_kernels_impl.h`
  - architecture dispatch macros
- `download_model.sh`
  - interactive small/large model downloader

## Build + Run

Build:
```bash
make blas
```

Smoke run:
```bash
./qwen_tts -d qwen3-tts-0.6b --text "Hello, how are you?" -o output.wav
```

Play output (macOS):
```bash
afplay output.wav
```

## Streaming Output (Future)

Streaming mode will generate audio chunks progressively, writing WAV data
as frames are decoded. The causal ConvNet decoder enables this without
lookahead. Not implemented in phase 1.

## Kernel/Optimization Rules

- Architecture dispatch is centralized in `qwen_tts_kernels_impl.h`.
- Keep generic/NEON/AVX variants functionally equivalent.
- If you optimize one path, verify no regression on others.
- Favor meaningful speedups; avoid complexity for tiny wins.

## Git Workflow

**New features MUST be developed on feature branches, NOT on main.**

- Branch naming: `feature/<short-name>` (e.g., `feature/streaming`, `feature/voice-clone`, `feature/http-server`)
- Open a PR to merge into main when ready
- Run `make test-all` before merging (both 0.6B and 1.7B must pass)
- Main branch must always build and pass tests

Roadmap and next steps are tracked in `PLAN.md`.

## Change Checklist For Agents

Before editing:
1. Identify behavioral contract impacted (CLI, output, speed, quality, memory).
2. Read corresponding source-of-truth file(s).
3. Create a feature branch if adding new functionality (NOT on main).

After editing:
1. Build: `make blas`
2. Run `make test-small` and `make test-large` (if model available).
3. Verify WAV output plays correctly and sounds reasonable.
4. Update `README.md` if CLI/runtime behavior changed.
5. Keep `PLAN.md` aligned if roadmap items completed or changed.

## Testing Rules (Comparative / A-B Tests)

When running comparative tests (normal vs streaming, before vs after optimization,
CLI vs server, etc.), **always use identical explicit parameters** across all runs:

- `--seed N` — same seed for every run
- `-s <speaker>` — explicit speaker name (not default)
- `-l <language>` — explicit language matching the text content
- Same `--text` across all runs
- Same model directory (`-d`)

**Default speaker for Italian and English**: `ryan` (`-s ryan`)

Example — comparing CLI normal vs streaming:
```bash
./qwen_tts -d qwen3-tts-0.6b --text "Test phrase here." --seed 42 -s ryan -l Italian -o normal.wav
./qwen_tts -d qwen3-tts-0.6b --text "Test phrase here." --seed 42 -s ryan -l Italian --stream -o stream.wav
```

For server tests, pass the same params in the JSON body:
```bash
curl -s http://localhost:8000/v1/tts -d '{"text":"Test phrase here.","seed":42,"speaker":"ryan","language":"Italian"}' -o server.wav
```

**Why**: Without explicit params, auto-detection or defaults may produce different
prompts (e.g. different codec token counts), making outputs non-comparable.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `qwen3-tts-0.6b/`, `qwen3-tts-1.7b/`
- `samples/`
- `TODO.md`
- virtualenv folders
