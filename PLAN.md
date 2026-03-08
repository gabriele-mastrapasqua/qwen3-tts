# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-07

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
  - Pre-transformer KV cache (8 layers, sliding window 72) — only process new frames
  - Cached latent output with windowed conv decoder (RF=20 frames context)
  - O(chunk_size) per streaming call instead of O(total_frames)
  - Bit-accurate vs full decode (correlation 1.000000, max diff 1 LSB)
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

- [ ] `[MED]` Implement speech tokenizer encoder (Mimi-based):
  - Conv encoder + 8-layer transformer + downsample + Split RVQ
  - Required for ICL mode (ref_text + ref_code)
  - NOT required for x_vector_only mode (which works now)

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
- [ ] `[MED]` ICL mode (ref_text + ref_code): requires speech tokenizer encoder (4.2)
- [x] `[MED]` `make test-clone` target (e2e: generate ref → clone → stream)

### 4.5 Reusable Voice Prompts

- [x] `[LOW]` `--save-voice <path>` — save speaker embedding to binary file
- [x] `[LOW]` `--load-voice <path>` — load pre-computed speaker embedding (skip extraction)

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

### 7.1 Metal GPU Offload (biggest remaining win)

- [ ] `[HIGH]` Metal compute shaders for bf16 matvec (Talker + CP)
  - Expected 3-5x speedup from GPU memory bandwidth
  - Only for macOS / Apple Silicon
- [ ] `[MED]` Metal for speech decoder convolutions

### 7.2 Further CPU Optimizations

- [ ] `[MED]` Profile 1.7B model bottlenecks (Talker prefill is slow: ~4s for 24 tokens)
- [ ] `[MED]` NEON/AVX snake activation kernels
- [ ] `[LOW]` Persistent BF16 KV cache (avoid bf16→f32 conversion)

---

## Phase 8: Windows Support (Optional)

**Goal**: Make it easy for Windows users to build and run. Keep the project clean —
don't add Windows-specific #ifdefs throughout the codebase if avoidable.

### Option A: WSL2 (Recommended)

WSL2 is the simplest path — our codebase is already Linux-compatible.

- [ ] `[MED]` Add Windows/WSL2 build instructions to README:
  ```
  # In WSL2 (Ubuntu):
  sudo apt install build-essential libopenblas-dev
  make blas
  ./qwen_tts_bin -d qwen3-tts-0.6b --text "Hello" -o output.wav
  ```
- [ ] `[MED]` Test full flow on WSL2: download → build → generate → play
- [ ] `[LOW]` Add WSL2 audio playback instructions (aplay, or copy WAV to Windows)

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

## Priority Summary

| Priority | Phase | Impact | Effort |
|----------|-------|--------|--------|
| **P0** | Phase 1: Instruct/Style | Enables emotion/style control (1.7B) | Low |
| **P1** | Phase 2: Streaming | Major UX — hear audio as it generates | Medium |
| **P2** | Phase 3: HTTP Server | Enables integration with external apps | Medium |
| **P3** | Phase 4: Voice Cloning | Major new capability — clone any voice | High |
| **P4** | Phase 5: VoiceDesign | Create custom voices from descriptions | Medium |
| **P5** | Phase 6: Quality | EOS reliability, reproducibility | Low |
| **P6** | Phase 7: Performance | Metal GPU, faster 1.7B | High |
| **P7** | Phase 8: Windows | Broader platform support | Low |

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
