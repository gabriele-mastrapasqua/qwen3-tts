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
  (2-row fused, 32 elem/iter, 8 NEON accumulators). x vector (4KB) fits in L1, weight access
  is sequential, HW prefetcher handles it. Bottleneck is memory bandwidth, not cache misses.
  *(2026-03-12)*

- [x] `[SKIP]` **Prefetch hints in CP loop**: Analyzed — can't prefetch 26MB/layer into L2
  (12MB). HW prefetcher handles sequential access within matvec. *(2026-03-12)*

- [x] `[LOW]` **Persist prefill buffers**: Prefill working buffers and f32 weight conversion
  buffers now persist in context across generations. Eliminates ~50MB of malloc/free traffic
  per generation in server mode. **Result: 38% faster on 2nd+ server request.** *(2026-03-12)*

- [x] `[MED]` **NEON-optimize speech decoder**: Replaced scalar RMSNorm (6 instances),
  scalar RoPE, and scalar attention with NEON-optimized versions. Added windowed causal
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

- [ ] `[MED]` **Batch text embedding with BLAS sgemm**: Instead of per-token matvec,
  collect all text token IDs, do a single bf16→f32 gather, then batch fc1 and fc2 as
  sgemm. For 50 tokens: 1 sgemm(50×2048 × 2048×2048) instead of 50 individual matvecs.
  Benefits both CLI and server. Less impactful now that LRU cache is active (warm hits
  are memcpy), but still helps CLI first-run and server cold call.

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

## Phase 10d: Batch Text Embedding (BLAS sgemm)

**Goal**: Replace per-token sequential matvec with a single batched sgemm for all text
tokens in the prefill. Currently each token does 2 individual bf16 matvec calls through
`embed_one_text_token()`. Batching them into one sgemm call lets BLAS optimize the
memory access pattern across all tokens simultaneously.

**Analysis**: For a 50-token prompt:
- Current: 50 × matvec(2048×2048) + 50 × matvec(1024×2048) = 100 individual calls
- Batched: 1 × sgemm(50×2048, 2048×2048) + SiLU + 1 × sgemm(50×2048, 2048×1024) = 2 calls
- BLAS sgemm is significantly more efficient than repeated matvec for batch sizes > ~8

**Interaction with LRU cache**: On server warm calls, most tokens are cache hits (memcpy).
The batch sgemm only helps for cache misses (first time a token is seen). Main benefit
is CLI mode and server cold calls where the LRU cache is empty.

**Complexity**: Low-medium (~80 lines). Gather bf16 embeddings for all tokens into a
contiguous f32 matrix, run 2 sgemm calls, scatter results. Need a temporary buffer
of `seq × text_hidden × sizeof(float)` (~400KB for 50 tokens).

**Risk**: LOW. Pure numerical change — output should be bit-identical to sequential
path (sgemm vs individual matvec may differ by FP rounding, but functionally equivalent).

### Tasks

- [ ] `[MED]` **Implement batched embedding function**
  - `embed_text_tokens_batch(ctx, token_ids[], n_tokens, output[])`
  - bf16→f32 gather → sgemm fc1 → SiLU → sgemm fc2 → bias add
  - Fall back to per-token for n_tokens < 4 (overhead not worth it)

- [ ] `[MED]` **Integrate into prefill path**
  - Replace the per-token loops in sections 0, 1, 3 of prompt construction
  - Keep LRU cache active — batch only the cache-miss tokens

---

## Phase 10e: Speculative Code Predictor Decoding (Experimental)

**Goal**: Reduce Code Predictor time (~55% of total) by predicting multiple codebook
tokens in parallel and verifying, rather than generating all 15 sequentially.

**⚠️ RISK: HIGH — may degrade audio quality. Must be thoroughly tested before merge.**

**Background**: The CP generates 15 codebook tokens per frame sequentially:
```
step 0: hidden → code[0]
step 1: embed(code[0]) → transformer → code[1]
step 2: embed(code[1]) → transformer → code[2]
...
```
Each step depends on the previous code. But the later codebooks encode finer details
(residual quantization — codebook 0 is coarse, codebook 15 is fine detail). This
means later codes have lower entropy and may be more predictable.

**Approach**: Draft-then-verify (inspired by speculative decoding in LLMs):
1. Run steps 0-4 normally (coarse codebooks, high information)
2. For steps 5-14: predict N codes in parallel using a lightweight draft head
3. Verify predictions against the real CP transformer
4. Accept correct predictions, recompute from first rejection point

**Estimated gain**: 20-30% of CP time IF acceptance rate is >60%. Each accepted
speculation saves one full transformer forward pass (~4ms).

**Complexity**: HIGH (~300-500 lines). Need to implement:
- Draft prediction head (small MLP trained on codebook statistics, or heuristic)
- Parallel evaluation of multiple candidate codes
- Verification loop with rollback
- Quality validation framework (correlation with non-speculative output)

**Risk**: HIGH. Wrong speculative codes = different audio. Even if individual codes
look similar, autoregressive error accumulation can cause quality drift. Must validate
with extensive listening tests and correlation metrics.

**Requirements before starting**:
- All other optimizations committed and tested
- Baseline audio samples saved for A/B comparison
- Correlation threshold defined (e.g., >0.999 vs non-speculative)
- Implemented behind `--speculative` flag, OFF by default

### Tasks

- [ ] `[LOW]` **Analyze codebook entropy by position**
  - Run 100+ generations, collect per-codebook token distributions
  - Measure conditional entropy: H(code[k] | code[0..k-1])
  - If later codebooks have low entropy, speculation is viable

- [ ] `[LOW]` **Implement draft-verify loop with flag**
  - `--speculative` CLI flag, disabled by default
  - Start conservative: speculate only codebooks 10-14
  - Measure acceptance rate and quality impact

- [ ] `[LOW]` **Quality validation**
  - Compare speculative vs non-speculative on 50+ test phrases
  - Correlation > 0.999 required for merge
  - Listening test: no audible artifacts

---

## Phase 10f: SDOT/SMMLA INT8 Native Dot Product (Optional, Architecture-Specific)

**Goal**: Use ARM NEON SDOT (`__ARM_FEATURE_DOTPROD`) instruction for native int8×int8
dot products in the Code Predictor matvec, bypassing f32 dequantization entirely.

**Context**: Current INT8 path dequantizes to f32 before multiply-accumulate (3 NEON ops).
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
| **P12** | Phase 10d: Batch Embedding | BLAS sgemm for text token projection | Low |
| **P13** | Phase 10e: Speculative CP | Draft-verify for later codebooks (⚠️ HIGH RISK) | High |
| **P14** | Phase 10f: SDOT INT8 | Native int8 dot product (optional, arch-specific) | Medium |
| **P15** | Phase 11: Metal GPU | FlashAttention Metal shader, MLX eval (optional, M3/M4+) | High |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
