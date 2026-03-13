# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-13

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
- [x] GitHub Actions CI/CD: build matrix, CodeQL, clang-tidy, ASan/UBSan, release artifacts

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

### 7.4 SIMD SiLU for SwiGLU (NEON tested, feat/labs) — TESTED, NO GAIN

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
- [x] `[MED]` SIMD int8→f32 matvec kernel (NEON/AVX, dequant on the fly, 2-row fused, multi-threaded)
- [x] `[MED]` Quality validation: same frame count (54), audio sounds correct
- [ ] `[LOW]` AVX equivalent for x86

**Result (Apple M1 8-core 16 GB, 4 threads):**

| Model | BF16 CP ms/f | INT8 CP ms/f | Speedup |
|-------|-------------|-------------|---------|
| 0.6B | 63.8 | 63.6 | 0% |
| 1.7B | 83.7 | 72.0 | ~14% |

**0.6B**: No measurable speedup. Same root cause as INT4: cp_hidden=1024 matrices
too small to be bandwidth-bound. INT8 dequant overhead (3 SIMD ops/4 weights on NEON,
similar on AVX) cancels the halved bandwidth vs BF16 dequant (1 op/4 weights).

**1.7B**: ~14% CP speedup (median of 5 runs). CP still has hidden=1024 but the larger
Talker creates more memory pressure, making bandwidth savings from INT8 more visible.
High variance due to 16GB RAM pressure with 3.6GB mmap'd model.
Note: Talker (hidden=2048) is NOT quantized — only CP.

Code is correct (same frame count, audio sounds good) and kept as `--int8` opt-in.

### 7.6 SDOT/I8MM Integer Dot Product Kernels (optional, future — needs M2+ or AVX-512 to test)

Current INT8 matvec dequantizes weights to f32 then does f32 FMA — 3 SIMD ops per
4 weights for dequant alone (NEON/AVX). Native integer dot product instructions can do int8×int8
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

## Phase 9: GitHub Actions CI/CD ✅ DONE

**Goal**: Automated cross-platform builds, benchmarks, and release artifacts.
Public repos get unlimited free CI minutes on GitHub-hosted runners.
**Status**: Core CI/CD implemented and running — build matrix, CodeQL, clang-tidy,
ASan/UBSan, release artifacts with checksums. Remaining items are stretch goals.

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
- [x] `[HIGH]` Install deps (`libopenblas-dev` on Linux, Xcode on macOS)
- [x] `[HIGH]` `make blas` — verify clean compile (zero warnings)
- [x] `[HIGH]` Binary exists and `./qwen_tts --help` returns 0

### 9.3 Benchmark + Hardware Dump

Each runner must dump system info and run a synthetic benchmark (no model needed):
- [x] `[MED]` Dump: OS version, CPU model, core count, RAM, BLAS version
- [ ] `[MED]` Micro-benchmark: matvec throughput (bf16, various sizes), to compare
  across runners without needing model files (models are too large for CI)
- [ ] `[LOW]` If feasible: download 0.6B model + run a fixed-seed generation and
  report ms/f breakdown. GitHub Actions has ~14GB RAM on Linux, enough for 0.6B (~3GB).
  Model download adds ~2 min but gives real end-to-end numbers.
- [ ] `[MED]` Upload benchmark results as workflow artifacts (JSON + human-readable summary)
- [ ] `[LOW]` Optional: commit results to a `benchmarks/` dir or publish to GitHub Pages

### 9.4 Release Artifacts

On manual trigger (or tag push like `v1.0`):
- [x] `[HIGH]` Build static binaries per platform/arch
- [x] `[HIGH]` Upload as GitHub Release assets (`qwen_tts-linux-x86_64`,
  `qwen_tts-linux-aarch64`, `qwen_tts-macos-arm64`, `qwen_tts-macos-x86_64`)
- [ ] `[MED]` Include version string in binary (`--version` flag)
- [x] `[LOW]` Checksums (SHA256) for all release binaries

### 9.5 Code Quality & Memory Safety (free, automated)

#### CodeQL (GitHub native)
- [x] `[HIGH]` Enable CodeQL for C/C++ (Settings → Code security → CodeQL)
  - Zero config: GitHub auto-detects `make blas` build
  - Finds: buffer overflows, use-after-free, integer overflow, null deref
  - Runs on every PR to main (free, ~5 min)
  - Particularly useful for our mmap + raw pointer + bf16 cast patterns

#### clang-tidy (static analysis)
- [x] `[MED]` Add `clang-tidy` job with `clang-analyzer-*` checks
  - Complementary to CodeQL (different analysis engine, finds different bugs)
  - Focus checks: `clang-analyzer-core.*`, `clang-analyzer-unix.*`,
    `clang-analyzer-security.*`, `bugprone-*`
  - Run on Linux runner only (clang-tidy pre-installed on ubuntu-latest)

#### ASan/UBSan (runtime memory safety)
- [x] `[HIGH]` CI job: `make debug` (ASan + UBSan) + run 2-3 fixed-seed generations
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

## Phase 10: CPU Cache & Memory Optimizations

**Goal**: Squeeze more performance from CPU-only inference through cache alignment,
memory layout, and allocation optimizations. Zero new dependencies.

**Context**: At RTF ~1.4–2.0 on M1, we're within 2–3x of an RTX 3090 running Python+PyTorch.
These optimizations target the remaining overhead from cache misses, unaligned allocations,
and per-token malloc traffic. Combined estimated gain: 10-25%.

**Allocation analysis (2026-03-12)**: Codebase is already well-optimized — zero malloc
in the Talker/CP decode loop. All decode buffers (`dec_x`, `dec_q`, etc.) are pre-allocated
at load time. The only per-token mallocs are in sampling (top-k/top-p work buffers, ~12KB)
and text embedding (~8KB during prompt). These are minor (<1-2ms total).
The real gains come from **cache alignment** of BLAS buffers and KV cache, not malloc elimination.

### 10.1 Quick Wins (high ROI, low effort)

- [x] `[HIGH]` **Cache-line align all hot-path buffers**: All decode buffers, KV caches, BLAS
  temporaries, and speech decoder buffers aligned to 64 bytes via `posix_memalign()`.
  Applied to talker, code predictor, and speech decoder. **Result: 24% total speedup**
  (prefill 84%, decoder 36%, CP 9%). Cross-platform (POSIX standard). *(2026-03-12)*

- [x] `[LOW]` **Pre-allocate sampling work buffers**: Reuse topk/topp work buffers across
  sample calls. Eliminates ~100 malloc/free pairs per generation. **Result: <1%.** *(2026-03-12)*

- [x] `[LOW]` **Pre-allocate text embedding temps**: Pre-allocate in context instead of
  per-call malloc. **Result: <1%.** *(2026-03-12)*

### 10.2 Memory Layout (medium effort, medium gain)

- [x] `[MED]` **Align KV cache to cache lines**: Done as part of 10.1 cache-line alignment.
  KV caches use `aligned_calloc(64, ...)` in talker and code predictor. *(2026-03-12)*

- [x] `[SKIP]` **Reorder context struct hot fields**: Analyzed — struct is 7.6KB (119 cache
  lines) but fields are pointer loads, not compute data. Hot decode pointers (kv_cache,
  dec_x, etc.) are already on adjacent cache lines 112-118. Reordering would not measurably
  help since the actual bottleneck is the data these pointers reference, not the pointer
  loads themselves. *(2026-03-12)*

- [x] `[SKIP]` **Reorder layer struct fields**: Same analysis — layer struct fields are
  weight pointers. The weight data locality matters, not the pointer ordering. *(2026-03-12)*

### 10.3 Advanced (lower priority)

- [x] `[SKIP]` **L1 cache blocking for matvec**: Analyzed — bf16 matvec kernel already optimal
  (2-row fused, 32 elem/iter, 8 SIMD accumulators NEON/AVX). x vector (4KB) fits in L1, weight access
  is sequential, HW prefetcher handles it. Bottleneck is memory bandwidth, not cache misses.
  *(2026-03-12)*

- [x] `[SKIP]` **Prefetch hints in CP loop**: Analyzed — can't prefetch 26MB/layer into L2
  (12MB). HW prefetcher handles sequential access within matvec. *(2026-03-12)*

- [x] `[LOW]` **Persist prefill buffers**: Prefill working buffers and f32 weight conversion
  buffers now persist in context across generations. Eliminates ~50MB of malloc/free traffic
  per generation in server mode. **Result: 38% faster on 2nd+ server request.** *(2026-03-12)*

- [x] `[MED]` **SIMD-optimize speech decoder**: Replaced scalar RMSNorm (6 instances),
  scalar RoPE, and scalar attention with SIMD-optimized versions (NEON/AVX). Added windowed causal
  attention kernel. Batched VQ dequant projections with BLAS sgemm. **Result: speech decoder
  ~11% faster (1446ms → 1288ms).** *(2026-03-12)*

---

## Phase 10b: Text Embedding Cache (Server Optimization)

**Goal**: Cache text token embeddings to avoid redundant matvec computation on repeated
or similar requests. Each `embed_one_text_token()` does 2 bf16 matvec ops (fc1: 2048×2048,
fc2: 1024×2048) — ~12M FLOPs per token. For a 60-token prompt that's ~29ms of pure compute
repeated identically on every request with the same text.

**Analysis (2026-03-12)**: Embedding cost is "Embed: 29ms" for long text (210 frames),
"Embed: 12ms" for short text. Not huge vs total RTF, but on server it's free savings.

### Tasks

- [x] `[HIGH]` **Cache special token embeddings at load time**: tts_pad, tts_bos, tts_eos
  computed once in `qwen_tts_load()`. Eliminates 3 × embed_one_text_token() per request.
  *(2026-03-12)*

- [x] `[MED]` **LRU hash map for text token embeddings**: Open-addressing hash map
  (2048 slots, ~8MB) with Knuth hash + linear probing + LRU eviction. Always active
  (CLI + server). **Result: 14% faster long-text server cold (RTF 1.55→1.33), best
  RTF 1.31 server warm.** Output bit-identical. *(2026-03-12)*

- [x] `[SKIP]` **Batch text embedding with BLAS sgemm**: Analyzed — "Embed: 37ms" on
  long text (210 frames, CLI) = 0.13% of total time. With LRU cache active, all tokens
  are cache hits on warm calls (memcpy). Even on cold CLI, embed is negligible vs
  Talker+CP (~25s). Batch sgemm would require bf16→f32 weight conversion buffers for
  text_projection (~32MB) for <0.2% gain. Not worth the complexity. *(2026-03-12)*

---

## Phase 10c: Decoder Thread Overlap (Pipeline Parallelism)

**Goal**: Overlap speech decoder execution with Talker+CP generation by running the
decoder in a separate thread. Currently the decoder runs AFTER all frames are generated,
wasting CPU cores that sit idle during generation.

**Analysis (2026-03-12)**: The speech decoder takes 2.4s (short) to 5.3s (long) and
runs sequentially after generation completes. During generation, the decode cores are
idle. The decoder is already designed as causal (sliding-window attention w=72, causal
ConvNet) and streaming mode already processes chunks of 10 frames incrementally.

**Architecture**: Producer-consumer pipeline with 2 threads:
- **Main thread** (producer): Runs Talker+CP generation loop, pushes completed frames
  to a shared queue
- **Decoder thread** (consumer): Pulls frames from queue, runs speech decoder
  incrementally using the existing `qwen_sd_stream_state_t` streaming infrastructure

**Implementation plan**:
1. Ringbuffer or simple mutex-guarded queue for codec frames (16 ints per frame)
2. Decoder thread blocks on condition variable when queue is empty
3. Main thread signals after each frame (or batch of N frames)
4. Decoder thread uses existing `qwen_speech_decoder_forward_streaming()` path
5. Main thread joins decoder thread after generation completes, collects final audio

**Estimated gain**: 15-20% of total pipeline time. The decoder runs ~25% of total time;
overlapping it with generation hides most of that cost. Gain is larger on long text
(more frames = more overlap opportunity, less edge effects from pipeline startup/drain).

**Complexity**: Medium (~200 lines). The decoder is already stateful and streaming-ready.
Main risk is thread synchronization correctness, but the data flow is simple (one
producer, one consumer, no shared mutable state except the queue).

**Risk**: LOW. The decoder is completely independent from Talker+CP — it only reads
completed codec frames and writes audio. No shared weights, no shared KV cache.
The streaming decoder path is already tested and produces identical output.

### Tasks

- [ ] `[HIGH]` **Implement decoder thread with frame queue**
  - pthread + mutex + condvar queue (or lockfree ringbuffer)
  - Decoder thread runs `qwen_speech_decoder_forward_streaming()` per chunk
  - Main thread pushes frames, signals decoder, joins at end
  - Collect audio samples from decoder thread after join

- [ ] `[HIGH]` **Verify bit-identical output**
  - Same seed must produce same WAV with and without threading
  - The streaming decoder path should already guarantee this

- [ ] `[MED]` **Benchmark CLI and server, short + long text**
  - Measure overlap efficiency: how much decoder time is hidden
  - Expected: better gains on long text (more pipeline overlap)

---

## Phase 10d: Batch Text Embedding (BLAS sgemm) — SKIPPED

**Status**: SKIPPED after analysis (2026-03-12).

**Reason**: Benchmarked "Embed: 37ms" on long text (210 frames, CLI) = 0.13% of total
pipeline time. With LRU cache active, all tokens are cache hits on warm calls (memcpy
instead of compute). Even on cold CLI, embedding is negligible vs Talker+CP (~25s total).

Batch sgemm would require bf16→f32 weight conversion buffers for text_projection (~32MB)
for <0.2% theoretical gain. Not worth the complexity.

---

## Phase 10e: Speculative Code Predictor Decoding — ABANDONED

**Status**: ABANDONED after thorough analysis and testing (2026-03-12).

**Analysis performed**:

1. **Codebook entropy analysis** (598 frames across 6 generations, 5 seeds + long text):
   - ALL 16 codebooks have high, uniform entropy: 7.6–8.5 bits (out of max 11 bits)
   - 266–439 unique values per codebook (out of 2048 possible)
   - Zero intra-frame correlation: CB[k] == CB[k-1] ≈ 0.0%
   - Frame-to-frame repetition: 1–9% (not exploitable)
   - Initial hypothesis was wrong: later codebooks are NOT more predictable
   - **Conclusion**: Draft+verify speculative decoding would have ~0% acceptance rate

2. **Early exit experiment** (fewer transformer layers for later codebooks):
   - Tested `--cp-early-exit N`: codebooks N–14 use 3 layers instead of 5
   - CP ms/f dropped 14% (80.8→69.8 for EE10)
   - **Critical discovery**: CP codes feed back into Talker via embedding sum
     (all 16 codebook embeddings are summed into the next frame's Talker input).
     Changing ANY CP code perturbs the entire generation trajectory.
   - Multi-seed stability test (7 seeds):
     - Baseline frame range: 70–103 (normal sampling variance)
     - EE8 frame range: 56–116 (doubled variance)
     - EE10 frame range: 79–161 (tripled variance, seed 456: 70→161 = 2.3x)
   - Audio quality: unpredictable — sometimes acceptable, sometimes garbled or 2x too long

3. **Root cause**: The model's feedback loop (CP codes → embedding sum → Talker input)
   makes ANY approximation in CP codes structurally unsafe. Small perturbations in
   later codebook tokens amplify exponentially through the autoregressive generation
   loop with sampling (temp=0.9). This is an inherent architectural property of
   Qwen3-TTS, not a fixable implementation issue.

**No variant of speculative, early-exit, or approximate CP is viable for this model.**

---

## Phase 10f: SDOT/SMMLA INT8 Native Dot Product (Optional, Architecture-Specific)

**Goal**: Use native int8×int8 dot product instructions (ARM SDOT / x86 VNNI) for
Code Predictor matvec, bypassing f32 dequantization entirely.

**Context**: Current INT8 path dequantizes to f32 before multiply-accumulate (3 SIMD ops).
SDOT computes 4 × int8 dot products in a single instruction, accumulating into int32.
This eliminates the dequant overhead that made INT8 neutral on 0.6B.

**Architecture requirements**:
- SDOT (`__ARM_FEATURE_DOTPROD`): Apple M1+, ARM Cortex-A76+
- SMMLA (`__ARM_FEATURE_MATMUL_INT8`): Apple M2+, ARM Cortex-X2+
- x86 equivalent: VNNI (AVX-512 VNNI on Ice Lake+, AVX-VNNI on Alder Lake+)

**Complexity**: Medium (~150 lines kernel + ~50 lines dispatch).
Need to quantize activations (x vector) to int8 too, not just weights. Per-vector
dynamic quantization adds ~0.1ms overhead per matvec call.

**Risk**: MEDIUM. Quantizing both weights AND activations to int8 introduces more
approximation error than weight-only quantization. May need per-channel scaling
or mixed precision (int8 weights × int8 activations with f32 accumulation + rescale).

**Priority**: LOW — architecture-specific optimization that breaks our goal of being
CPU-agnostic. Implement as opt-in behind `--sdot` flag with runtime feature detection.
The main codebase should remain portable (bf16 matvec as default path).

### Tasks

- [ ] `[LOW]` **Runtime feature detection**: Check `__ARM_FEATURE_DOTPROD` at compile
  time, `getauxval(AT_HWCAP)` at runtime on Linux, `sysctlbyname` on macOS.
- [ ] `[LOW]` **SDOT int8 matvec kernel**: int8 weights × int8 activations → int32 accum → f32 rescale
- [ ] `[LOW]` **Quality validation**: Compare audio output with bf16 baseline

---

## Phase 10g: Scalar Hot Paths → SIMD (NEON/AVX) / Algorithm Optimization

**Goal**: Vectorize remaining scalar C loops on hot paths and fix algorithmic
inefficiencies (O(n²) sorts). These are the last major CPU-only wins before
hitting memory bandwidth limits.

**Context (2026-03-13)**: Profiling identified several hot paths that are 100%
scalar C — no SIMD, no BLAS. The sampling pipeline is the worst offender:
called ~8000× per generation with O(n²) sorting. Speech decoder has scalar
depthwise conv and per-timestep LayerNorm. Streaming mode also has a structural
overhead from blocking the main thread during decode.

**Approach**: One task at a time, test after each with identical params
(`--seed 42 -s ryan -l Italian`, same text). Ordered by: (1) expected gain,
(2) low cognitive difficulty, (3) low code invasion risk.

---

### 10g.1 Top-k: Replace Selection Sort with Quickselect

**What**: `sampling.c:55-70` — O(k×n) selection sort to find k-th largest value.
Default top_k=50, vocab=2048 → 102k comparisons per call × ~8000 calls/gen.

**Fix**: nth_element / quickselect (O(n) average). Find the k-th value, then
threshold in a single pass. No SIMD needed, pure algorithm improvement.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — ~30 lines, drop-in replacement |
| Risk | LOW — same output (deterministic threshold, identical filtering) |
| Code invasion | LOW — contained in `qwen_tts_sampling.c` only |
| Expected gain | **2-3× faster top-k**, ~1-2% total frame time |
| Calls/gen | ~8000 (1 per token + 15 per CP frame) |

---

### 10g.2 Softmax: SIMD Vectorized exp + Horizontal Sum (NEON/AVX)

**What**: `sampling.c:34-44` — 3 scalar passes over vocab (2048): find max,
compute `expf()` + sum, normalize. Called ~8000×/gen. 100% scalar.

**Fix**: SIMD 4-wide exp approximation (NEON `vexpq_f32` / AVX `_mm256_exp_ps`,
or vDSP on macOS) + horizontal reduction. Keep generic fallback.

| Metric | Value |
|--------|-------|
| Difficulty | MEDIUM — SIMD exp approximation needs care for numerical stability |
| Risk | LOW-MEDIUM — exp approximation error must not change sampling distribution meaningfully |
| Code invasion | LOW — contained in `qwen_tts_sampling.c`, dispatch via arch ifdef |
| Expected gain | **2-4× faster softmax**, ~1-2% total frame time |
| Calls/gen | ~8000 |

**Note**: On macOS, `vvexpf()` from Accelerate is already fast with `-ffast-math`.
Benchmark before implementing custom SIMD exp to see if there's actual headroom.

---

### 10g.3 Top-p: Replace Full Sort with Partial Sort / Early Exit

**What**: `sampling.c:73-102` — O(n²) selection sort on FULL vocab to compute
cumulative probability. With default top_p=1.0 this is skipped, but any top_p<1.0
triggers a full sort of 2048 elements per call.

**Fix**: Use partial quicksort or heapsort that stops once cumsum > p. Typical
top_p=0.9 only needs the top ~50-100 tokens sorted, not all 2048.

| Metric | Value |
|--------|-------|
| Difficulty | MEDIUM — quicksort with early exit, ~50 lines |
| Risk | LOW — same output (cumulative threshold is identical) |
| Code invasion | LOW — contained in `qwen_tts_sampling.c` |
| Expected gain | **5-10× faster top-p** when top_p<1.0, 0% when top_p=1.0 (skipped) |
| Calls/gen | ~8000 (when enabled) |

---

### 10g.4 Speech Decoder: SIMD Depthwise Conv (k=7) (NEON/AVX)

**What**: `speech_decoder.c:971-981` — scalar depthwise conv, 1024 channels ×
~10k samples × kernel 7. Called in 2 ConvNeXt blocks per decode.

**Fix**: SIMD vectorize the inner kernel loop (7 multiply-accumulates). Process
4 channels simultaneously (NEON `vfmaq_f32` / AVX `_mm256_fmadd_ps`).

| Metric | Value |
|--------|-------|
| Difficulty | MEDIUM — fixed kernel size (7) simplifies SIMD, ~40 lines per arch |
| Risk | LOW — pure computation, easy to verify bit-exact |
| Code invasion | LOW — add NEON/AVX path alongside scalar in speech_decoder.c |
| Expected gain | **1.5-2× faster depthwise conv**, ~2-3% of decoder time |
| Calls/gen | 2 blocks × per-decode |

---

### 10g.5 Speech Decoder: SIMD LayerNorm Per-Timestep (NEON/AVX)

**What**: `speech_decoder.c:986-999` — scalar LayerNorm across 1024 channels per
timestep. 2 passes (sum+sum_sq, normalize). Called ~16k× per decode.

**Fix**: SIMD horizontal sum reduction + broadcast multiply. Process 4-8 channels
per iteration (NEON `vaddq_f32` / AVX `_mm256_add_ps`).

| Metric | Value |
|--------|-------|
| Difficulty | MEDIUM — similar to existing SIMD RMSNorm, ~30 lines per arch |
| Risk | LOW — pure computation, easy to verify |
| Code invasion | LOW — add NEON/AVX path in speech_decoder.c |
| Expected gain | **2-3× faster LayerNorm**, ~1-2% of decoder time |
| Calls/gen | ~16k timesteps |

---

### 10g.6 Streaming: Pipeline Parallelism (Decoder Thread)

**What**: Streaming mode (`--stream`) calls the speech decoder synchronously in
the main thread, blocking Talker+CP generation. Normal mode already uses a
decoder thread for pipeline overlap. Streaming gets RTF ~2.0 vs ~1.4 normal.

**Fix**: Use the existing decoder thread infrastructure in streaming mode too.
The decoder thread calls the audio callback instead of accumulating to buffer.
Main thread never blocks on decode.

| Metric | Value |
|--------|-------|
| Difficulty | HIGH — thread synchronization with callback, edge cases at drain |
| Risk | MEDIUM — threading bugs, callback from wrong thread, audio ordering |
| Code invasion | MEDIUM — modify `qwen_tts.c` decoder thread + streaming paths (~50-100 lines) |
| Expected gain | **Streaming RTF from ~2.0 to ~1.4** (match normal mode) |

**Note**: This overlaps with Phase 10c (decoder thread). If 10c is implemented
first, streaming just needs to hook into it. If not, implement both together.

---

### 10g.7 Separate INT8 Fields from `qwen_cp_layer_t`

**What**: `qwen_cp_layer_t` is 264 bytes (33 pointers) — half are optional INT8
weight pointers that waste cache when `--int8` is not used.

**Fix**: Move INT8 fields to a separate `qwen_cp_layer_int8_t` struct, allocated
only when `--int8` is enabled. Reduces `qwen_cp_layer_t` to ~132 bytes.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — mechanical refactor, move fields + update accessors |
| Risk | LOW — no behavioral change, just memory layout |
| Code invasion | LOW-MEDIUM — touch `qwen_tts.h` + `qwen_tts_code_predictor.c` |
| Expected gain | **~2-3% less cache pollution** in CP loop when INT8 disabled |
| Calls/gen | N/A (structural) |

---

### Priority Order (do one at a time, test after each)

| Order | Task | Difficulty | Risk | Expected Gain |
|-------|------|------------|------|---------------|
| 1 | 10g.1 Top-k quickselect | LOW | LOW | 2-3× top-k |
| 2 | ~~10g.7 Separate INT8 from CP layer~~ | LOW | LOW | SKIP (see below) |
| 3 | ~~10g.2 Softmax SIMD (NEON/AVX)~~ | MEDIUM | LOW-MED | SKIP (see below) |
| 4 | 10g.3 Top-p partial sort | MEDIUM | LOW | 5-10× top-p |
| 5 | ~~10g.4 Depthwise conv SIMD (NEON/AVX)~~ | MEDIUM | LOW | SKIP (see below) |
| 6 | ~~10g.5 LayerNorm SIMD (NEON/AVX)~~ | MEDIUM | LOW | SKIP (see below) |
| 7 | **10g.6 Streaming pipeline** | HIGH | MEDIUM | **DONE: RTF 2.0→1.38** |

### Results & Analysis (2026-03-13)

**10g.1 Top-k quickselect**: DONE. Selection sort O(k×n) → quickselect O(n).
Codec head+sampling: 93ms → 21-24ms (**4× faster**). Output bit-identical.

**10g.7 Separate INT8 from CP layer**: SKIP. Only 5 layers × 264B = 1.3KB.
Removing INT8 fields saves ~660B but the bottleneck is weight data, not pointer
loads. Code churn not worth ~2-3% theoretical cache improvement.

**10g.2 Softmax SIMD (NEON/AVX)**: SKIP. Post-quickselect, total sampling is ~21ms/101 frames
= 0.2ms/frame. Softmax is ~1.5ms of that (101 × 3072 expf). With `-ffast-math`,
the compiler already vectorizes `expf` via platform libraries. No headroom.

**10g.3 Top-p partial sort**: Low priority. Default top_p=1.0 already skips the
sort entirely. Only triggered if user explicitly sets `--top-p 0.9` etc. Keep as
optional future improvement.

**10g.4 Depthwise conv SIMD + 10g.5 LayerNorm SIMD (NEON/AVX)**: SKIP. The speech decoder
runs in a decoder thread overlapped with generation. Decoder finishes BEFORE
Talker+CP completes (9.9s decoder vs 10.8s generation). It's NOT the bottleneck.
ConvNeXt depthwise (1.4M FLOPs) is 600× less compute than BLAS-accelerated PW1
(838M FLOPs). These scalar loops are negligible in the total decoder time.

**10g.6 Streaming pipeline**: DONE. Decoder thread now runs in streaming mode too,
calling audio_cb from the thread. Main thread never blocks on decode.
**Streaming RTF: 2.04 → 1.38** (matches normal mode). Output bit-identical across
all 4 modes (CLI normal, CLI stream, server normal, server stream). *(2026-03-13)*

---

## Phase 10h: INT8 Talker Quantization (1.7B Priority)

**Goal**: Extend `--int8` to quantize Talker weights in addition to CP. The 1.7B Talker
has 2.8 GB of BF16 weights (28 layers × ~100MB/layer) that are bandwidth-bound during
single-token decode. INT8 halves this to 1.4 GB, potentially saving 2-3 seconds per
generation on the 1.7B model.

**Motivation**: Micro-benchmarks show scalar loop optimizations (fast sigmoid, NEON GELU)
save <50ms total — negligible on a 27s generation. The real bottleneck is memory bandwidth
for matvec. INT8 directly attacks this: half the data = half the bandwidth.

**Context**: INT8 CP on 1.7B gave 14% speedup (86→72 ms/f). The Talker has 4x larger
matrices (hidden=2048, inter=6144 vs CP hidden=1024, inter=1024), making it MORE
bandwidth-bound and MORE likely to benefit from INT8.

**Why 1.7B benefits more than 0.6B**: On 0.6B, INT8 CP gave 0% speedup because
hidden=1024 matrices are too small to be bandwidth-bound — the INT8 dequant overhead
(3 SIMD ops) cancels the bandwidth savings. On 1.7B, the same CP (hidden=1024) gained
14% because the larger Talker model creates more memory pressure. The 1.7B Talker
itself (hidden=2048) has matrices large enough that bandwidth is clearly the bottleneck.

### Tasks

#### 10h.1 Add INT8 fields to `qwen_talker_layer_t` struct

**What**: Add INT8 weight pointers and per-row scales to the Talker layer struct,
mirroring the existing CP pattern.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — copy pattern from `qwen_cp_layer_t` |
| Risk | NONE — struct fields only, no behavioral change |
| Files | `qwen_tts.h` |

#### 10h.2 Quantize Talker weights at load time

**What**: After loading BF16 weights in `qwen_talker_load()`, call
`qwen_quantize_bf16_to_int8()` for each layer's QKV, output, gate+up fused, and
down projections. Guard behind `ctx->use_int8`.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — same API as CP quantization |
| Risk | LOW — quantization is well-tested |
| Files | `qwen_tts_talker.c` |

#### 10h.3 Dispatch INT8 matvec in Talker single-token decode

**What**: In `qwen_talker_step()`, add `if (l->wq_int8)` checks for QKV, output,
gate+up, and down projections. Same dispatch pattern as CP.

Note: Talker prefill uses BLAS `cblas_sgemm` (batch matmul), NOT matvec. INT8 batch
matmul is not implemented and would require a different kernel. **Keep prefill as BF16
BLAS** — prefill is a one-time cost and already fast.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — copy dispatch pattern from CP |
| Risk | MEDIUM — quality impact on Talker (it drives the generation) |
| Files | `qwen_tts_talker.c` |

#### 10h.4 Quantize codec head (optional)

**What**: The codec head maps Talker hidden → codec vocab (3072 tokens). Currently
BF16 matvec. Could be INT8 + argmax fused (like CP lm_head).

| Metric | Value |
|--------|-------|
| Difficulty | LOW — same fused argmax pattern |
| Risk | LOW — codec head is small, and argmax is robust to quantization |
| Files | `qwen_tts.c` (codec head projection) |

#### 10h.5 Test quality and performance

**What**: Run same-seed A/B tests:
- `./qwen_tts -d qwen3-tts-1.7b --text "..." -s ryan -l Italian --seed 42`
- Same with `--int8`
- Compare: frame count, RTF, listen to both

| Metric | Value |
|--------|-------|
| Expected 1.7B Talker speedup | 15-30% (120→85-100 ms/f) |
| Expected 1.7B total saving | 2-4 seconds on long text |
| Expected 0.6B impact | Minimal (same as CP: ~0%) |
| Quality risk | Talker is autoregressive — INT8 error accumulates |

### Priority Order

| Order | Task | Difficulty | Risk |
|-------|------|------------|------|
| 1 | 10h.1 Struct fields | LOW | NONE |
| 2 | 10h.2 Load-time quantization | LOW | LOW |
| 3 | 10h.3 Decode dispatch | LOW | MEDIUM |
| 4 | 10h.5 Test quality + perf | — | — |
| 5 | 10h.4 Codec head (optional) | LOW | LOW |

---

## Phase 10i: INT4 (Q4_0) Talker Quantization (1.7B Only) — NO SPEEDUP

**Status**: IMPLEMENTED but NO SPEEDUP. INT4 Talker is ~4% SLOWER than BF16 on 1.7B
(82.6 vs 79.3 ms/f). The Q4 nibble unpacking overhead outweighs bandwidth savings,
same root cause as INT4 CP on 0.6B. INT8 remains the best quantization for 1.7B (15% speedup).

**A/B Results (Italian, seed=42, ryan, Apple M1 16GB)**:
| Config | Frames | Talker ms/f | CP ms/f | Total | RTF |
|--------|--------|-------------|---------|-------|-----|
| 0.6B BF16 | 38 | 22.5 | 82.0 | 6.5s | 2.15 |
| 1.7B BF16 | 38 | 79.3 | 87.0 | 13.1s | 4.32 |
| 1.7B INT8 | 39 | 67.4 | 78.7 | 11.2s | 3.59 |
| 1.7B INT4 | 39 | 82.6 | 81.7 | 14.1s | 4.51 |

Audio quality: all four outputs sound good. Frame count variation (38 vs 39) is
normal with quantization-induced sampling divergence.

**Conclusion**: Keep `--int4` as opt-in flag but don't recommend it. INT8 is the
sweet spot for 1.7B Talker. For maximum speed, use 0.6B (RTF 2.15 vs 3.59).

**Original Goal**: Quantize Talker weights to 4-bit (Q4_0 format) for the 1.7B model. Reduces
weight memory from 2.8 GB (BF16) to ~0.7 GB, halving INT8's 1.4 GB. Expected 25-30%
Talker speedup over BF16 with much less memory pressure than INT8.

**Format**: Q4_0 — groups of 32 weights packed into 16 bytes (nibble pairs) + 1 fp16
scale factor = 18 bytes per 32 weights. Same format used by llama.cpp and whisper.cpp.

**Why 1.7B only**: On 0.6B, INT8 already showed 0% speedup (hidden=1024 too small to
be bandwidth-bound). INT4's extra dequantization overhead would make it SLOWER on 0.6B.
The 1.7B Talker (hidden=2048, inter=6144) is clearly bandwidth-bound and benefits from
reduced data movement.

**Memory comparison**:
| Format | Talker weights | Total model | RAM headroom (16GB) |
|--------|---------------|-------------|---------------------|
| BF16   | 2.8 GB        | 3.6 GB      | ~12 GB              |
| INT8   | 1.4 GB + 2.8 GB mmap | 4.2 GB peak | ~12 GB (but mmap pages compete) |
| INT4   | 0.7 GB + 2.8 GB mmap | 3.5 GB peak | ~12 GB (mmap pages paged out faster) |

**Quality risk**: INT4 has 16x fewer representable values than INT8. Talker is
autoregressive — quantization error accumulates across frames. Must test quality
carefully with listening comparison.

### Tasks

#### 10i.1 Implement Q4_0 quantize and matvec kernels

**What**: Add Q4_0 data structures and two core functions:
- `qwen_quantize_bf16_to_q4_0()`: Pack BF16 weights into Q4_0 blocks (32 weights → 18 bytes)
- `qwen_matvec_q4_0()`: NEON matvec with Q4_0 weights (unpack nibbles → int8 → f32 → FMA)
- `qwen_matvec_q4_0_qkv()`: Fused QKV variant

Q4_0 block layout (18 bytes):
```
struct { float16 scale; int8_t qs[16]; } // 16 bytes store 32 nibbles
```

NEON dequant per 32 weights: load 16 bytes → split nibbles (AND 0x0F, SHR 4) →
subtract 8 (unsigned→signed) → widen to int16 → widen to int32 → cvt to f32 →
multiply by scale → FMA with input vector.

| Metric | Value |
|--------|-------|
| Difficulty | MEDIUM — nibble packing/unpacking is fiddly |
| Risk | LOW — well-known format, many reference implementations |
| Files | `qwen_tts_kernels_neon.c`, `qwen_tts_kernels_generic.c`, `qwen_tts_kernels.c` |

#### 10i.2 Add INT4 fields to Talker layer struct + CLI flag

**What**: Add Q4_0 weight pointers to `qwen_talker_layer_t`. Add `--int4` CLI flag
and `use_int4` field to context. INT4 and INT8 are mutually exclusive.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — struct fields + flag parsing |
| Risk | NONE — no behavioral change |
| Files | `qwen_tts.h`, `main.c` |

#### 10i.3 Quantize Talker to Q4_0 at load time

**What**: In `qwen_talker_load()`, when `use_int4` is set, call
`qwen_quantize_bf16_to_q4_0()` for all Talker weight matrices. Sequential INT4
prefill (same approach as INT8 — one token at a time via `qwen_talker_step()`).

| Metric | Value |
|--------|-------|
| Difficulty | LOW — same pattern as INT8 |
| Risk | LOW |
| Files | `qwen_tts_talker.c`, `qwen_tts.c` |

#### 10i.4 Dispatch Q4_0 matvec in Talker decode

**What**: In `qwen_talker_step()`, add `if (l->wq_q4)` checks before INT8 and BF16
fallbacks. Priority: INT4 > INT8 > BF16.

| Metric | Value |
|--------|-------|
| Difficulty | LOW — copy dispatch pattern |
| Risk | MEDIUM — quality impact |
| Files | `qwen_tts_talker.c` |

#### 10i.5 A/B Comparison: 0.6B BF16 vs 1.7B INT4

**What**: Compare full pipeline quality and performance:
- `./qwen_tts -d qwen3-tts-0.6b --text "Ciao, come stai oggi? Spero tutto bene." -s ryan -l Italian --seed 42 -o /tmp/06b_full.wav`
- `./qwen_tts -d qwen3-tts-1.7b --text "Ciao, come stai oggi? Spero tutto bene." -s ryan -l Italian --seed 42 --int4 -o /tmp/17b_int4.wav`
- Compare: frame count, ms/f Talker, ms/f CP, total time, RTF
- Listen to both WAVs for quality assessment
- Also test English: "Hello, how are you doing today?"

| Metric | Value |
|--------|-------|
| Expected 1.7B INT4 Talker speedup | 25-30% over BF16 |
| Expected memory savings | 0.7 GB vs 1.4 GB (INT8) vs 2.8 GB (BF16) |
| Quality risk | HIGH — must verify with listening test |

### Priority Order

| Order | Task | Difficulty | Risk |
|-------|------|------------|------|
| 1 | 10i.1 Q4_0 kernels | MEDIUM | LOW |
| 2 | 10i.2 Struct + CLI | LOW | NONE |
| 3 | 10i.3 Load-time quant | LOW | LOW |
| 4 | 10i.4 Decode dispatch | LOW | MEDIUM |
| 5 | 10i.5 A/B comparison | — | — |

---

## Phase 10j: Batch Vectorized Math + Embedding SIMD

**Goal**: Squeeze remaining per-frame overhead from scalar math and BF16→F32
conversion loops. Two independent optimizations, both cross-platform (NEON/AVX/generic).

**Inspiration**: antirez/qwen-asr uses Apple Accelerate `vvexpf()` for batch
exponential in SwiGLU, avoiding per-element `expf()` overhead.

### Tasks

#### 10j.1 Batch vvexpf for SwiGLU (Talker + CP)

**What**: Restructure the SwiGLU loop to batch all gate values, compute exp in
one call, then apply sigmoid×up. On macOS use `vvexpf()` (Accelerate/vForce);
on Linux/generic use a scalar loop (compiler vectorizes with `-ffast-math`).

Current (scalar, per-element):
```c
for (int o = 0; o < inter; o++) {
    float g = gate[2*o], u = gate[2*o+1];
    gate[o] = g / (1.0f + expf(-g)) * u;  // expf called N times
}
```

Proposed (batched):
```c
// 1. Extract gate values into contiguous buffer, negate
for (int o = 0; o < inter; o++) neg_g[o] = -gate[2*o];
// 2. Batch exp (vvexpf on macOS, scalar loop elsewhere)
vvexpf(exp_g, neg_g, &inter);  // or: for(...) exp_g[o] = expf(neg_g[o]);
// 3. Apply sigmoid × up
for (int o = 0; o < inter; o++) {
    float g = gate[2*o], u = gate[2*o+1];
    gate[o] = g / (1.0f + exp_g[o]) * u;
}
```

**Call count**: Talker 28 layers × 3072 + CP 5×15 × 1024 = ~163K expf/frame.
**Platform support**:
- macOS: `vvexpf()` from `<Accelerate/Accelerate.h>` (already linked)
- Linux/generic: scalar loop, compiler auto-vectorizes with `-ffast-math`
- AVX/SSE: no manual intrinsics needed (compiler handles it)

| Metric | Value |
|--------|-------|
| Expected gain | 1-3 ms/frame (vvexpf is 4-8x faster than scalar expf) |
| Difficulty | LOW — restructure loop + conditional vvexpf call |
| Risk | NONE — mathematically identical, no approximation |
| Files | `qwen_tts_talker.c`, `qwen_tts_code_predictor.c` |

#### 10j.2 NEON/AVX Codec Embedding BF16→F32 Accumulation

**What**: Vectorize the scalar loop that accumulates 15 codebook embeddings per
frame. Currently 15 × hidden BF16→F32 conversions done one element at a time.

Current (scalar):
```c
for (int j = 0; j < h; j++) step_embed[j] += bf16_to_f32(emb[j]);
```

Proposed (NEON):
```c
for (int j = 0; j + 7 < h; j += 8) {
    uint16x8_t bf = vld1q_u16(emb + j);
    float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16));
    float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16));
    vst1q_f32(step_embed + j,     vaddq_f32(vld1q_f32(step_embed + j), f0));
    vst1q_f32(step_embed + j + 4, vaddq_f32(vld1q_f32(step_embed + j + 4), f1));
}
```

AVX equivalent: `_mm256_slli_epi32(_mm256_cvtepu16_epi32(load128), 16)` → cast to f32.

**Call count**: 15 codebooks × 1024 elements = 15K conversions per frame.
**Platform support**:
- ARM NEON: `vshll_n_u16` + `vaddq_f32`
- x86 AVX2: `_mm256_cvtepu16_epi32` + `_mm256_slli_epi32` + `_mm256_add_ps`
- Generic: existing scalar loop (unchanged)

| Metric | Value |
|--------|-------|
| Expected gain | 0.5-1 ms/frame |
| Difficulty | LOW — straightforward NEON/AVX conversion |
| Risk | NONE — pure arithmetic optimization |
| Files | `qwen_tts.c` (generation loop, codec embedding accumulation) |

#### 10j.3 Delta Prefill / KV Cache Reuse (Server Mode) — DONE

**What**: In server mode, compare the current prompt's input embeddings with the previous
request's. If the prefix matches (same speaker, language, ChatML header), skip
re-prefilling those tokens and reuse the existing KV cache entries.

Typical prompt structure (~16 tokens for 0.6B):
- Tokens 0-7: ChatML header + speaker + language + codec prefix (IDENTICAL across requests with same speaker)
- Tokens 8-15: Text content + eos + final (DIFFERENT per request)

If speaker/language match, skip prefill for first ~8 tokens = ~50% prefill savings.

**Implementation**:
- Store previous `input_embeds` + length in `ctx->prev_input_embeds` / `ctx->prev_prefill_len`
- On new request, compare embedding vectors (memcmp per position) to find longest common prefix
- Set `ctx->kv_len = delta_start` (reuse cached KV entries for matched prefix)
- Process only delta tokens via sequential `qwen_talker_step()` (correct for all quant modes)
- First call uses full BLAS batch prefill (BF16 mode) or sequential (INT8/INT4)
- Causal attention guarantees: prefix tokens produce identical KV regardless of following text

| Metric | Value |
|--------|-------|
| Expected gain | ~50% prefill time on repeated speaker |
| Difficulty | MEDIUM — need to track previous state, handle edge cases |
| Risk | LOW — automatic, transparent optimization; falls back to full prefill on mismatch |
| Files | `qwen_tts.c` (prefill section), `qwen_tts.h` (context fields) |

### Priority Order

| Order | Task | Expected Gain | Difficulty | Risk |
|-------|------|---------------|------------|------|
| 1 | 10j.1 Batch vvexpf SwiGLU | 1-3 ms/f | LOW | NONE |
| 2 | 10j.2 Codec embedding NEON/AVX | 0.5-1 ms/f | LOW | NONE |
| 3 | 10j.3 Delta prefill (server) | ~50% prefill | MEDIUM | LOW | **DONE** |

---

## Phase 11: GPU Acceleration via Metal (Optional, Future)

**Goal**: Evaluate Metal GPU offload for Apple Silicon, specifically for attention and
matmul kernels. Pure C + Objective-C wrapper (`.m` files), no external dependencies.
Metal framework ships with macOS — it's a system framework, not an external lib.

**Context**: We previously implemented and removed a Metal backend (PR #14) because
it was 1.3x SLOWER than CPU on M1 due to per-dispatch command buffer overhead
(~50μs × 112 dispatches/step). However:
- Metal 4 (2025) introduces tensors as first-class citizens and 4.7x transformer kernel speedups
- FlashAttention on GPU fuses QKV projection + attention + softmax into a single kernel,
  eliminating the per-dispatch overhead that killed our previous attempt
- M3/M4 have improved GPU memory bandwidth and Metal 4 support

### 11.1 FlashAttention Metal Kernel

FlashAttention fuses the entire attention computation (Q×K^T, softmax, ×V) into a
single GPU kernel, avoiding materialization of the N×N attention matrix. This is what
gives the 30-40% speedup reported in GPU benchmarks.

**Complexity assessment:**
- ~300-500 lines Metal shader code (tiled attention with online softmax)
- ~200-300 lines Objective-C wrapper (command buffer, pipeline state, dispatch)
- Requires careful tile size tuning for Apple GPU architecture
- KV cache must live on GPU (unified memory makes this transparent)
- **Medium-high complexity**, but no external dependencies

**When to attempt:** Only when M3/M4 hardware is available for testing.
Metal 4's tensor APIs may simplify the shader significantly.

- [ ] `[LOW]` Prototype fused attention Metal shader (FlashAttention-style)
- [ ] `[LOW]` Benchmark vs CPU on M3/M4 (must beat CPU to justify inclusion)
- [ ] `[LOW]` If faster: add `--gpu` flag with automatic CPU fallback

### 11.2 MLX Integration (Alternative)

Apple's [MLX](https://github.com/ml-explore/mlx) framework provides optimized GPU
kernels for Apple Silicon. Antirez uses MLX in [flux2.c](https://github.com/antirez/flux2)
without external dependencies by linking the MLX C API directly.

**Trade-offs vs raw Metal:**
- Pro: Pre-optimized matmul, attention, and elementwise kernels
- Pro: Unified memory management, lazy evaluation, JIT compilation
- Con: MLX is a separate library (not a system framework like Metal)
- Con: Adds ~50MB dependency, requires C++ runtime

**Not recommended** unless MLX becomes a system framework on macOS.
Raw Metal with custom shaders keeps the zero-dependency philosophy.

- [ ] `[LOW]` Evaluate MLX C API for matmul/attention offload
- [ ] `[LOW]` Compare MLX vs raw Metal performance on M3/M4

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
| **P9** | Phase 10: CPU Cache | Cache alignment, memory layout, alloc optimization | Medium |
| **P10** | Phase 10b: Embedding Cache | LRU token embedding cache (server RTF 1.31) | Low |
| **P11** | Phase 10c: Decoder Thread | Pipeline overlap generation+decode (est. 15-20%) | Medium |
| **P12** | Phase 10d: Batch Embedding | ~~SKIPPED~~ — 0.13% of pipeline, not worth it | — |
| **P13** | Phase 10e: Speculative CP | ~~ABANDONED~~ — codebook feedback loop makes it structurally unsafe | — |
| **P14** | Phase 10f: SDOT INT8 | Native int8 dot product (optional, arch-specific) | Medium |
| **P15** | Phase 10g: Scalar→SIMD+Algo | Sampling vectorization, depthwise SIMD (NEON/AVX), streaming pipeline | Medium |
| **P16** | Phase 11: Metal GPU | FlashAttention Metal shader, MLX eval (optional, M3/M4+) | High |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
