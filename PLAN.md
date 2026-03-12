# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-12

Core engine is **COMPLETE** and producing good audio for both 0.6B and 1.7B models.
This document tracks all planned features, improvements, and known issues.

---

## DONE

- [x] Full pipeline: Talker → Code Predictor → Speech Decoder → WAV
- [x] Both model sizes: 0.6B (hidden=1024) and 1.7B (hidden=2048)
- [x] Standard HuggingFace safetensors loader (mmap, JSON header parsing)
- [x] Speech tokenizer loaded from separate `speech_tokenizer/model.safetensors`
- [x] 1.7B MTP projection (small_to_mtp_projection bridging talker→CP hidden)
- [x] Config auto-detect (nested JSON blanking for correct parsing)
- [x] 9 preset speakers, 10 languages, multilingual
- [x] BLAS acceleration (Accelerate/OpenBLAS), NEON/AVX SIMD kernels
- [x] Makefile test targets: `test-small`, `test-large`, `test-regression`, `test-all`
- [x] `download_model.sh` for both model sizes
- [x] ~0.5-0.7x realtime (0.6B), ~0.2-0.4x realtime (1.7B) on Apple Silicon

---

## Phase 1: Instruct / Style Control (1.7B only)

**Background**: Qwen3-TTS CustomVoice 1.7B supports an `instruct` parameter that controls
speaking style, emotion, and prosody via natural language. The 0.6B model does NOT support
instruct (the Python code explicitly skips it for `0b6` models).

The instruct text is injected into the ChatML prompt as a user turn:
```
<|im_start|>user\n{instruct}<|im_end|>\n
```
This is tokenized and prepended to the generation prompt before the assistant turn.

### Tasks

- [x] `[HIGH]` Add `--instruct <text>` CLI flag
  - Only effective with 1.7B model; warn/ignore for 0.6B
  - Example: `--instruct "Speak in an angry tone"`
  - Example: `--instruct "Speak slowly and softly"`
  - Example: `--instruct "用特别愤怒的语气说"` (Chinese style instructions work too)
- [x] `[HIGH]` Implement instruct prompt injection in `qwen_tts.c`:
  - Tokenize instruct text as `<|im_start|>user\n{instruct}<|im_end|>\n`
  - Insert these tokens into the prefill sequence before the assistant turn
  - The rest of the prompt (role + codec + text) remains unchanged
- [x] `[MED]` Add `make test-instruct` target to validate instruct works on 1.7B
- [x] `[MED]` Update README with instruct examples and speaker descriptions:

**Known speaker characteristics** (from HuggingFace model card):

| Speaker | Description | Best Language |
|---------|-------------|---------------|
| Vivian | Bright young female | Chinese |
| Serena | Warm, gentle female | Chinese |
| Uncle_Fu | Seasoned male, mellow | Chinese |
| Dylan | Youthful Beijing male | Chinese (Beijing dialect) |
| Eric | Lively Sichuan male | Chinese (Sichuan dialect) |
| Ryan | Dynamic male with rhythm | English |
| Aiden | Sunny American male | English |
| Ono_Anna | Playful Japanese female | Japanese |
| Sohee | Warm Korean female | Korean |

> Note: Most speakers are optimized for their native language. Ryan is the best
> choice for English and Italian. Chinese-native speakers (Dylan, Uncle_Fu, Eric)
> may produce artifacts or elongated vowels in non-Chinese languages.

---

## Phase 2: Streaming Output

**Background**: The speech decoder is fully causal (no lookahead), so chunked decoding
is architecturally possible. Qwen3-TTS paper describes a "Dual-Track" streaming arch
where audio is generated as soon as text tokens arrive (97ms first-packet latency in Python).

For our C engine, streaming means: generate N frames → decode chunk → write/output audio
immediately, instead of waiting for full generation to complete.

Reference: qwen-asr implements streaming via callbacks (`qwen_set_token_callback`) and
chunked processing (`qwen_transcribe_stream`). We can use a similar pattern.

### 2.1 Chunked Generation + WAV Streaming

- [x] `[HIGH]` Implement `--stream` CLI flag:
  - Generate frames in chunks (e.g., 10 frames = 0.8s audio per chunk)
  - Decode each chunk through speech decoder immediately
  - Write WAV header with unknown length, update at end
  - First audio heard within ~1-2 seconds of starting
- [x] `[MED]` Speech decoder incremental decode (optimization):
  - Pre-transformer KV cache (8 layers, sliding window 72) + cached latent output
  - Windowed conv decoder (RF=20 frames context) for O(chunk_size) per call
  - O(chunk_size) per streaming call instead of O(total_frames)
  - Bit-accurate: correlation 1.000000, max diff 1 LSB vs full decode
- [x] `[MED]` Configurable chunk size: `--stream-chunk <frames>` (default: 10)

### 2.2 Raw PCM to stdout

- [x] `[MED]` `--stdout` flag: output raw s16le 24kHz mono PCM to stdout
  ```bash
  ./qwen_tts -d model --text "Hello" --stdout | aplay -f S16_LE -r 24000 -c 1
  # macOS: ... --stdout | play -t raw -r 24000 -e signed -b 16 -c 1 -
  ```
- [x] `[MED]` `--stdout` implies `--stream` and forces silent mode (stderr only)

### 2.3 Callback API (for embedding)

- [x] `[LOW]` Add `qwen_tts_set_audio_callback(ctx, fn, userdata)`:
  - Called with each decoded audio chunk (float* samples, int n_samples)
  - Enables embedding in other applications without file I/O

---

## Phase 3: HTTP Server / REST API

**Background**: Multiple community projects already wrap Qwen3-TTS Python in FastAPI
servers (Qwen3-TTS_server, Qwen3-TTS-Openai-Fastapi). Our C engine can offer a
lightweight, zero-dependency alternative.

### 3.1 Minimal HTTP Server

- [x] `[HIGH]` Implement embedded HTTP server (single-threaded, no external deps):
  - Minimal HTTP/1.1 parser (enough for POST + headers)
  - Listen on configurable port: `--serve <port>` (default: 8080)
  - JSON request/response using simple hand-rolled parser
  - Model loaded once at startup, shared across requests
- [x] `[HIGH]` POST `/v1/tts` endpoint:
  ```json
  {
    "text": "Hello world",
    "speaker": "ryan",
    "language": "English",
    "instruct": "Speak cheerfully",
    "temperature": 0.9,
    "top_k": 50
  }
  ```
  Response: WAV file (Content-Type: audio/wav)
- [x] `[MED]` POST `/v1/tts/stream` endpoint:
  - Same request format
  - Response: chunked transfer encoding with raw PCM
  - Client receives audio progressively as it's generated
- [x] `[MED]` GET `/v1/speakers` — list available speakers
- [x] `[MED]` GET `/v1/health` — status check

### 3.2 OpenAI-Compatible API (optional)

- [x] `[LOW]` POST `/v1/audio/speech` (OpenAI TTS API format):
  ```json
  {
    "model": "qwen-tts",
    "input": "Hello world",
    "voice": "ryan",
    "response_format": "wav",
    "speed": 1.0
  }
  ```
  - Enables drop-in replacement for OpenAI TTS in existing apps
  - Map `voice` to speaker names, `speed` to instruct hints

### 3.3 Makefile Targets

- [x] `[MED]` `make serve` — build + start server on default port
- [x] `[MED]` `make test-serve` — start server, send test request, validate response

---

## Phase 4: Voice Cloning (Base models)

**Background**: Qwen3-TTS Base models (not CustomVoice) support voice cloning from
3 seconds of reference audio. The process:

1. **Speech Tokenizer Encoder**: Encode reference audio → codec tokens (ref_code)
2. **Speaker Encoder**: Extract mel spectrogram → speaker embedding (x-vector)
3. **Prompt injection**: ref_code + speaker embedding + target text → Talker

The Base model has additional components not present in CustomVoice:
- **Speaker Encoder** (ECAPA-TDNN style): mel spectrogram → speaker embedding
- The speech tokenizer encoder (inverse of decoder) is the same tokenizer model

Two modes:
- **ICL mode** (default): Uses both ref_code and speaker embedding. Requires ref_text transcript.
- **x_vector_only mode**: Uses only speaker embedding. Lower quality but no transcript needed.

### 4.1 Download Base Models

- [x] `[HIGH]` Extend `download_model.sh` for Base model variants:
  - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (Base 0.6B)
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Base 1.7B)
  - Speaker encoder weights are in the main model.safetensors (76 tensors)

### 4.2 Speech Tokenizer Encoder (for ICL mode)

- [x] `[MED]` Implement speech tokenizer encoder (Mimi-based):
  - Conv encoder (4 stages: ResBlock + ELU + stride conv, rates [4,5,6,8])
  - 8-layer transformer (LayerNorm, GELU MLP, sliding window 250, NeoX RoPE)
  - Downsample conv (stride=2, k=4)
  - Split RVQ: separate semantic (1 codebook) + acoustic (15 codebooks) quantizers
  - Encoder-specific input projections loaded from speech_tokenizer safetensors
  - Codebooks shared with decoder (verified bit-identical)
  - Note: 0.6B Base model ICL mode generates very short output (~2 frames);
    same behavior in Python reference. x_vector_only mode works well.

### 4.3 Speaker Encoder

- [x] `[HIGH]` Implement ECAPA-TDNN speaker encoder:
  - WAV reader (16/32-bit PCM, mono/stereo)
  - Mel spectrogram extraction (n_fft=1024, n_mels=128, hop=256, sr=24kHz, slaney norm)
  - Full ECAPA-TDNN: initial TDNN → 3× SE-Res2Net blocks → MFA → ASP → FC
  - Output: 1024-dim speaker embedding
  - Verified bit-exact match with Python reference
- [x] `[MED]` Reference audio loading: WAV file → float32 samples (24kHz required)

### 4.4 Voice Clone Pipeline

- [x] `[HIGH]` Implement x_vector_only voice clone prompt format:
  - Extract speaker embedding via ECAPA-TDNN speaker encoder
  - Inject speaker embedding into codec prefix (replaces discrete speaker token)
  - Base model auto-detected from config (`tts_model_type: "base"`)
- [x] `[MED]` CLI flags:
  - `--ref-audio <path.wav>` — reference audio file
  - `--xvector-only` — use speaker embedding only (default when no --ref-text)
- [x] `[MED]` ICL mode (ref_text + ref_code):
  - Speech encoder encodes ref audio → 16 codebook codes per frame
  - ICL prompt: ref_text + target_text + EOS paired with codec_pad,
    then codec_bos + ref_code embeddings (sum of 16 codebook lookups) paired with tts_pad
  - Matches Python reference implementation behavior exactly
- [x] `[MED]` `make test-clone` target (e2e: generate ref → clone → stream)

### 4.5 Reusable Voice Prompts

- [x] `[LOW]` `--save-voice <path>` — save speaker embedding to binary file
- [x] `[LOW]` `--load-voice <path>` — load pre-computed speaker embedding (skip extraction)

### 4.6 Voice Clone Demo (Makefile)

- [x] `[MED]` `make demo-clone` target:
  - Accepts custom reference audio: `make demo-clone REF=my_voice.wav`
  - Supports WAV/OGG/MP3 input (any format the pipeline accepts)
  - Generates English + Italian cloned samples to `samples/`
  - Customizable text via `TEXT=` and `TEXT_IT=` variables

---

## Phase 5: VoiceDesign

**Background**: Qwen3-TTS-12Hz-1.7B-VoiceDesign accepts natural language descriptions
to create entirely new voices without reference audio. Examples:
- "A deep male voice with a British accent, speaking slowly"
- "Young energetic female, cheerful and fast-paced"
- "萝莉女声，撒娇稚嫩" (young girl voice, cute and naive)

The VoiceDesign model uses the same `instruct` prompt mechanism but generates a novel
timbre from the description instead of using a preset speaker.

### Tasks

- [x] `[MED]` Extend `download_model.sh` for VoiceDesign model:
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- [x] `[MED]` Implement VoiceDesign prompt format:
  - Same as CustomVoice but without speaker token in codec prefix
  - Auto-detected from config (empty `spk_id`)
  - Instruct describes the desired voice characteristics
- [x] `[MED]` CLI: `--voice-design` flag (auto-detected, or force with flag)
- [x] `[LOW]` Add `make test-voice-design` target

---

## Phase 6: Quality & Reliability

### 6.1 EOS Reliability

- [x] `[MED]` Add `--max-duration <seconds>` flag:
  - Convert to max frames: seconds × 12.5
  - Default: no limit (use max_new_tokens=8192)
  - Prevents runaway generation with bad seeds
- [x] `[LOW]` EOS boosting after expected duration (gentle logit boost at 2x expected frames)

### 6.2 Seed / Reproducibility

- [x] `[LOW]` Add `--seed <n>` flag for reproducible generation
- [x] `[LOW]` Document seed behavior and output variability (README section)

---

## Phase 7: Performance

### 7.1 Metal GPU Offload — REMOVED

Metal GPU backend was implemented and benchmarked but turned out ~1.3x **slower**
than the optimized NEON CPU path on Apple Silicon (unified memory = shared bandwidth
ceiling). The entire Metal backend was removed as dead code in commit history.

- [x] ~~Metal compute shaders~~ — removed (slower than CPU)
- [x] ~~Full GPU transformer step~~ — removed (still slower than CPU)
- [ ] `[LOW]` CUDA/HIP backend stubs (for future NVIDIA/AMD support)

### 7.2 Generation Speed Variability

The same text with different seeds (or same seed with different temperatures) produces
audio of very different durations — up to 3-7x range. This is **inherent model behavior**
(confirmed identical in Python). The Talker's autoregressive loop decides when to emit EOS,
and sampling randomness (temp=0.9 default) heavily influences frame count.

- Greedy (temp=0, top_k=1) produces consistent, short output
- temp=0.9 + top_k=50 (defaults) gives natural-sounding but variable-length output
- Lower temperature (0.6-0.7) is a good middle ground: less variability, still natural
- **No single "sweet spot"** — it's a quality/consistency tradeoff the user should control
- All sampling params remain user-configurable (--temperature, --top-k, --top-p, --seed)

### 7.3 Further CPU Optimizations

- [x] `[MED]` Profile 1.7B model bottlenecks:
  - Per-frame timing added to generation loop (Talker step, CP, embed, codec head)
  - **1.7B breakdown** (45 frames, Apple Silicon M-series, 4 threads):
    - Prefill: 4218ms (169ms/tok) — 3.1× slower than 0.6B (55ms/tok)
    - Talker step: 92.2ms/f — 3.9× slower (hidden 2048 vs 1024)
    - Code Predictor: 74.9ms/f — ~same (both cp_hidden=1024)
    - Speech decoder: 56ms/f — ~same
    - **Total: 167ms/frame → 0.48× realtime** (need 80ms/f for 1.0×)
  - **0.6B breakdown** (61 frames): Talker 23.6ms/f + CP 69.6ms/f = 93ms/f → 0.86× realtime
  - **Conclusion**: CP is the bottleneck for both models (15 sequential passes)
- [x] `[MED]` NEON/AVX snake activation kernels:
  - macOS: vDSP_vsmul → vvsinf → vDSP_vsq → vDSP_vsma (Accelerate)
  - ARM NEON: 4-wide SIMD with scalar sinf per lane + fma
  - Generic fallback for non-SIMD platforms
- [x] `[LOW]` Persistent BF16 KV cache (halves KV memory, bf16 stored/read in attention)

### 7.4 NEON SiLU for SwiGLU (feat/labs) — TESTED, NO GAIN

SwiGLU activation (`silu(gate) * up`) uses scalar `expf()` in the hot loop.
Called per-layer per-step: Talker (28 layers × 3072) + CP (5 layers × 15 passes × 3072)
= ~316K scalar expf calls per frame.

**Result**: No measurable speedup. SiLU is <1% of frame time (~0.2ms out of 80ms).
With `-ffast-math`, clang already uses Apple Accelerate's optimized `expf`.
The Schraudolph NEON approximation also caused autoregressive divergence
(different frame count with same seed due to ~1e-4 error accumulating).

- [x] `[MED]` ~~Implement NEON fused SiLU×up kernel~~ — tested, reverted (no gain + divergence)
- [ ] ~~AVX equivalent~~ — not worth it
- [ ] ~~Benchmark~~ — done: 0% speedup, quality risk

### 7.5 INT8 Quantization for Code Predictor (feat/labs)

CP is 55% of total time and NOT fully bandwidth-bound at hidden=1024.
INT4 failed (unpack overhead > bandwidth gain on small matrices), but INT8 has
much simpler unpack (vmovl_s8, 1 op per 8 weights) and halves bandwidth vs BF16.

**Design**: BF16 remains the default. `--int8` CLI flag enables runtime quantization
of CP weights (bf16→int8 at load time with per-row absmax scaling). This lets users
choose quality (BF16) vs speed (INT8) without separate model files. The Talker stays
BF16 regardless — it's only 35% of frame time and more sensitive to precision.

- [x] `[HIGH]` `--int8` CLI flag: quantize CP weights at load time (bf16→int8 per-row absmax)
- [x] `[MED]` NEON int8→f32 matvec kernel (dequant on the fly, 2-row fused, multi-threaded)
- [x] `[MED]` Quality validation: same frame count (54), audio sounds correct
- [ ] `[LOW]` AVX equivalent for x86

**Result (Apple M1 8-core 16 GB, 4 threads):**

| Model | BF16 CP ms/f | INT8 CP ms/f | Speedup |
|-------|-------------|-------------|---------|
| 0.6B | 63.8 | 63.6 | 0% |
| 1.7B | 83.7 | 72.0 | ~14% |

**0.6B**: No measurable speedup. Same root cause as INT4: cp_hidden=1024 matrices
too small to be bandwidth-bound. INT8 dequant overhead (3 NEON ops/4 weights) cancels
the halved bandwidth vs BF16 dequant (1 NEON op/4 weights via vshll).

**1.7B**: ~14% CP speedup (median of 5 runs). CP still has hidden=1024 but the larger
Talker creates more memory pressure, making bandwidth savings from INT8 more visible.
High variance due to 16GB RAM pressure with 3.6GB mmap'd model.
Note: Talker (hidden=2048) is NOT quantized — only CP.

Code is correct (same frame count, audio sounds good) and kept as `--int8` opt-in.

### 7.6 SDOT/I8MM Integer Dot Product Kernels (optional, future — needs M2+ or AVX-512 to test)

Current INT8 matvec dequantizes weights to f32 then does f32 FMA — 3 NEON ops per
4 weights for dequant alone. Native integer dot product instructions can do int8×int8
multiplication directly in hardware, avoiding the f32 conversion entirely.

**Available instructions (compile-time detection via preprocessor macros):**

| Instruction | Macro | Min Arch | Apple Silicon | Op |
|-------------|-------|----------|---------------|----|
| SDOT | `__ARM_FEATURE_DOTPROD` | ARMv8.2 | M1+ | 4× int8·int8 → int32, 1 cycle |
| SMMLA | `__ARM_FEATURE_MATMUL_INT8` | ARMv8.6 | M2+ | 8× int8·int8 → int32, 1 cycle |
| VNNI | `__AVX512VNNI__` / `__AVXVNNI__` | AVX-512/AVX2 | N/A | 4× int8·int8 → int32 |

**Design**: Quantize BOTH weights (int8, per-row absmax) AND activations (int8,
per-tensor dynamic quantization of x vector before each matvec). Then use SDOT/SMMLA
to compute dot products entirely in int8→int32, dequantize only the final accumulator.

**Compile-time dispatch**: Use `#ifdef __ARM_FEATURE_DOTPROD` etc. to select the
optimal kernel at compile time. With `-march=native`, the compiler defines the right
macros automatically for the target CPU. For portable binaries, use runtime detection
(`getauxval(AT_HWCAP)` on Linux, `sysctl` on macOS) with function pointers.

**Status**: Optional/deferred — M1 has SDOT but not SMMLA; no M2+ or AVX-512
hardware available to test. Propose when hardware becomes available, do not auto-implement.

- [ ] `[LOW]` SDOT matvec kernel: int8 weights × int8 activations → int32 accumulate
- [ ] `[LOW]` Dynamic per-tensor int8 quantization of activation vector x
- [ ] `[LOW]` SMMLA kernel for M2+ (2× throughput vs SDOT)
- [ ] `[LOW]` AVX VNNI kernel for x86
- [ ] `[LOW]` Runtime CPU feature detection for portable binaries

---

## Phase 8: Windows Support (Optional)

**Goal**: Make it easy for Windows users to build and run. Keep the project clean —
don't add Windows-specific #ifdefs throughout the codebase if avoidable.

### Option A: WSL2 (Recommended)

WSL2 is the simplest path — our codebase is already Linux-compatible.

- [x] `[MED]` Add Windows/WSL2 build instructions to README:
  ```
  # In WSL2 (Ubuntu):
  sudo apt install build-essential libopenblas-dev
  make blas
  ./qwen_tts_bin -d qwen3-tts-0.6b --text "Hello" -o output.wav
  ```
- [ ] `[MED]` Test full flow on WSL2: download → build → generate → play (no Windows machine available; marked beta)
- [x] `[LOW]` Add WSL2 audio playback instructions (aplay, or copy WAV to Windows)

### Option B: Native MSVC/MinGW (Lower priority)

- [ ] `[LOW]` Evaluate MinGW-w64 + OpenBLAS build feasibility
  - mmap → need Windows MapViewOfFile wrapper
  - pthreads → need Windows threads or pthreads-w64
  - Heavy #ifdef burden — only if there's real demand
- [ ] `[LOW]` Alternative: CMake + vcpkg for Windows native build

### Decision

WSL2 is the recommended path. Native Windows only if users specifically request it.
The README already mentions WSL2 as the recommended approach.

---

## Phase 9: GitHub Actions CI/CD

**Goal**: Automated cross-platform builds, benchmarks, and release artifacts.
Public repos get unlimited free CI minutes on GitHub-hosted runners.

### 9.1 Build Matrix

Trigger: **manual only** (`workflow_dispatch` button in Actions tab) on `main` branch.
No automatic runs on every push — we decide when to launch.

Target matrix:

| Runner | OS | Arch | BLAS | Notes |
|--------|-----|------|------|-------|
| `ubuntu-latest` | Linux | x86_64 | OpenBLAS | Primary Linux target |
| `ubuntu-24.04-arm` | Linux | aarch64 | OpenBLAS | ARM Linux (free for public repos) |
| `macos-latest` | macOS | ARM64 (M-series) | Accelerate | Apple Silicon |
| `macos-13` | macOS | x86_64 | Accelerate | Intel Mac |
| `windows-latest` | Windows/WSL2 | x86_64 | OpenBLAS | Via WSL2 Ubuntu layer |

Cross-compilation (stretch goal):
- Linux x86 → Linux ARM via `aarch64-linux-gnu-gcc` + OpenBLAS ARM
- Useful if native ARM runners are unavailable

### 9.2 Build Verification

Each runner must:
- [ ] `[HIGH]` Install deps (`libopenblas-dev` on Linux, Xcode on macOS)
- [ ] `[HIGH]` `make blas` — verify clean compile (zero warnings)
- [ ] `[HIGH]` Binary exists and `./qwen_tts --help` returns 0

### 9.3 Benchmark + Hardware Dump

Each runner must dump system info and run a synthetic benchmark (no model needed):
- [ ] `[MED]` Dump: OS version, CPU model, core count, RAM, BLAS version
- [ ] `[MED]` Micro-benchmark: matvec throughput (bf16, various sizes), to compare
  across runners without needing model files (models are too large for CI)
- [ ] `[LOW]` If feasible: download 0.6B model + run a fixed-seed generation and
  report ms/f breakdown. GitHub Actions has ~14GB RAM on Linux, enough for 0.6B (~3GB).
  Model download adds ~2 min but gives real end-to-end numbers.
- [ ] `[MED]` Upload benchmark results as workflow artifacts (JSON + human-readable summary)
- [ ] `[LOW]` Optional: commit results to a `benchmarks/` dir or publish to GitHub Pages

### 9.4 Release Artifacts

On manual trigger (or tag push like `v1.0`):
- [ ] `[HIGH]` Build static binaries per platform/arch
- [ ] `[HIGH]` Upload as GitHub Release assets (`qwen_tts-linux-x86_64`,
  `qwen_tts-linux-aarch64`, `qwen_tts-macos-arm64`, `qwen_tts-macos-x86_64`)
- [ ] `[MED]` Include version string in binary (`--version` flag)
- [ ] `[LOW]` Checksums (SHA256) for all release binaries

### 9.5 Code Quality & Memory Safety (free, automated)

#### CodeQL (GitHub native)
- [ ] `[HIGH]` Enable CodeQL for C/C++ (Settings → Code security → CodeQL)
  - Zero config: GitHub auto-detects `make blas` build
  - Finds: buffer overflows, use-after-free, integer overflow, null deref
  - Runs on every PR to main (free, ~5 min)
  - Particularly useful for our mmap + raw pointer + bf16 cast patterns

#### clang-tidy (static analysis)
- [ ] `[MED]` Add `clang-tidy` job with `clang-analyzer-*` checks
  - Complementary to CodeQL (different analysis engine, finds different bugs)
  - Focus checks: `clang-analyzer-core.*`, `clang-analyzer-unix.*`,
    `clang-analyzer-security.*`, `bugprone-*`
  - Run on Linux runner only (clang-tidy pre-installed on ubuntu-latest)

#### ASan/UBSan (runtime memory safety)
- [ ] `[HIGH]` CI job: `make debug` (ASan + UBSan) + run 2-3 fixed-seed generations
  - Short English phrase (seed 42) + longer multilingual phrase (seed 7)
  - Catches: out-of-bounds, stack overflow, undefined behavior, alignment issues
  - Needs 0.6B model download (~2 min) but catches real bugs in real paths
  - ASan is 2x native speed — fast enough for CI (vs Valgrind at 20x)
  - Run on Linux x86_64 only (ASan is best supported there)

> **Why no Valgrind?** ASan covers 95% of memory bugs and runs 10x faster.
> Valgrind nightly would add maintenance for marginal gain on a one-man project.
> If a subtle leak surfaces that ASan misses, add Valgrind as a one-off debug step.

### 9.6 Free Multi-CPU Testing Services (stretch)

For public open-source repos, explore:
- **GitHub-hosted runners**: Already cover x86_64 + ARM64 for Linux/macOS (free)
- **Actuated.dev**: ARM runners for GitHub Actions (free tier for OSS)
- **Cirrus CI**: Free for public repos, offers ARM64 Linux + FreeBSD
- **builds.sr.ht**: Free for OSS, has various arch (amd64, arm64, riscv64)

These would let us test on diverse CPUs (Graviton, Ampere, RISC-V) without hardware.

### 9.7 Workflow Design

```yaml
# .github/workflows/build.yml
name: Build & Benchmark
on:
  workflow_dispatch:        # Manual "Run workflow" button
    inputs:
      create_release:
        description: 'Create GitHub Release with binaries'
        type: boolean
        default: false
  push:
    tags: ['v*']            # Also trigger on version tags
```

Key design decisions:
- **Manual trigger only** for build+bench (no auto-run on push to main)
- Tag push (`v1.0`, `v1.1`) also triggers and creates a release automatically
- `create_release` checkbox in manual trigger to optionally create release
- Each job uploads its binary + benchmark JSON as artifacts
- Final job collects all artifacts into a release (if requested)

Separate workflows:
- `.github/workflows/build.yml` — manual build matrix + bench + release
- `.github/workflows/codeql.yml` — auto on PR to main (CodeQL + clang-tidy)
- `.github/workflows/safety.yml` — auto on PR to main (ASan/UBSan test run)

---

## Priority Summary

| Priority | Phase | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | Phase 1: Instruct/Style | Enables emotion/style control (1.7B) | Low |
| **P1** | Phase 2: Streaming | Major UX — hear audio as it generates | Medium |
| **P2** | Phase 3: HTTP Server | Enables integration with external apps | Medium |
| **P3** | Phase 4: Voice Cloning | Major new capability — clone any voice | High |
| **P4** | Phase 5: VoiceDesign | Create custom voices from descriptions | Medium |
| **P5** | Phase 6: Quality | EOS reliability, reproducibility | Low |
| **P6** | Phase 7: Performance | CPU optimizations, faster 1.7B | High |
| **P7** | Phase 8: Windows | Broader platform support | Low |
| **P8** | Phase 9: GH Actions | Cross-platform CI, benchmarks, releases | Medium |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
