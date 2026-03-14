# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-13

Core engine is **COMPLETE** and producing good audio for both 0.6B and 1.7B models.
This document tracks completed work and remaining future ideas.

---

## COMPLETED

All major features and optimizations have been implemented and verified:

### Features
- [x] Full pipeline: Talker → Code Predictor → Speech Decoder → WAV
- [x] Both model sizes: 0.6B (hidden=1024) and 1.7B (hidden=2048)
- [x] Standard HuggingFace safetensors loader (mmap, JSON header parsing)
- [x] 9 preset speakers, 10 languages, multilingual
- [x] Instruct / Style Control (`--instruct`, 1.7B only)
- [x] Streaming output (`--stream`, `--stdout`, audio callback API)
- [x] HTTP Server / REST API (`--serve`, OpenAI-compatible `/v1/audio/speech`)
- [x] Voice Cloning (Base models): ICL mode + x-vector only, `--ref-audio`, `--save-voice`/`--load-voice`
- [x] VoiceDesign (1.7B-VoiceDesign model): custom voice from text description
- [x] Quality: `--max-duration`, `--seed`, EOS boosting
- [x] `download_model.sh` for all model variants (CustomVoice, Base, VoiceDesign)
- [x] Makefile test suite: `test-small`, `test-large`, `test-regression`, `test-all`, `test-clone`, etc.
- [x] WSL2 build instructions in README

### Performance Optimizations
- [x] BLAS acceleration (Accelerate/OpenBLAS), NEON/AVX SIMD kernels
- [x] Cache-line aligned buffers (24% total speedup)
- [x] LRU text embedding cache (server RTF 1.31)
- [x] Decoder thread overlap (pipeline parallelism — decoder runs concurrent with generation)
- [x] Streaming pipeline parallelism (RTF 2.0→1.38)
- [x] Multi-row bf16 matvec (2-row fused), Unified QKV dispatch, Fused gate+up
- [x] NEON RMSNorm, NEON attention (dot+V accum), NEON RoPE
- [x] Fused argmax_matvec in CP, CP allocation elimination
- [x] Top-k quickselect (4x faster sampling)
- [x] Batch vvexpf SwiGLU (Talker + CP)
- [x] NEON/AVX BF16→F32 codec embedding accumulation
- [x] Delta prefill / KV cache reuse (server mode, ~50% prefill savings)
- [x] Persistent prefill buffers (38% faster 2nd+ server request)
- [x] SIMD speech decoder (RMSNorm, RoPE, attention, windowed causal attention)
- [x] BF16 KV cache (halves KV memory)
- [x] INT8 quantization (`--int8`): ~14% CP speedup on 1.7B
- [x] INT4 Q4_0 quantization (`--int4`): implemented but no speedup (kept as opt-in)

### CI/CD
- [x] GitHub Actions: build matrix (Linux x86/ARM, macOS ARM/x86), CodeQL, clang-tidy, ASan/UBSan
- [x] Release artifacts with checksums on tag push

### Performance Summary (Apple M1 8-core 16 GB, 4 threads)

| Model | Talker ms/f | CP ms/f | RTF | Notes |
|-------|-------------|---------|-----|-------|
| 0.6B BF16 | ~26 | ~78 | 1.4-1.6 | Best speed |
| 1.7B BF16 | ~79 | ~87 | 3.5-4.3 | Best quality |
| 1.7B INT8 | ~67 | ~79 | 3.0-3.6 | Recommended for 1.7B |

### Experiments That Didn't Work
- **Metal GPU**: 1.3x SLOWER than CPU on M1 (unified memory = shared bandwidth ceiling). Removed.
- **NEON SiLU**: 0% speedup (SiLU is <1% of frame time). `-ffast-math` already optimizes expf.
- **INT4 Q4_0**: 4% SLOWER on 1.7B (nibble unpack overhead > bandwidth savings). Kept as opt-in.
- **INT8 on 0.6B**: 0% speedup (hidden=1024 too small to be bandwidth-bound).
- **Speculative CP decoding**: ABANDONED (codebook feedback loop makes it structurally unsafe).
- **Batch text embedding (BLAS sgemm)**: SKIPPED (0.13% of pipeline, not worth it).
- **Softmax SIMD**: SKIPPED (post-quickselect, sampling is 0.2ms/frame).
- **Depthwise conv SIMD / LayerNorm SIMD**: SKIPPED (decoder runs overlapped, not bottleneck).

---

## FUTURE IDEAS (not currently planned)

### Phase 12: Reusable Custom Voices from Voice Clone

**Goal**: Enable creating persistent, reusable custom voices from a voice clone operation.
Currently `--save-voice` saves the 1024-dim speaker embedding (x-vector), but the full
voice clone quality comes from ICL mode which also uses reference codec tokens (ref_code).
The idea is to save BOTH the speaker embedding AND the codec tokens (or even the full
KV cache prefix from prefill) in a reusable format, so subsequent generations with
different text can reproduce the same cloned voice without re-processing the reference audio.

**Motivation**: Extend the voice palette beyond the 9 preset speakers. Users could clone
any voice once, save it as a "voice profile", and reuse it across many different texts.
This effectively turns Qwen3-TTS into a system with unlimited custom voices.

**What needs to be saved** (to investigate):
1. **Speaker embedding (x-vector)** — already saved via `--save-voice` (1024 floats, ~4KB)
2. **Reference codec tokens (ref_code)** — the 16-codebook tokens from encoding the reference audio.
   These are used in ICL mode for in-context learning. Without them, only x-vector mode works
   (lower quality). Saving ref_code would enable full ICL quality on reload.
3. **Reference text transcript (ref_text)** — needed for ICL mode prompt construction.
   Could be stored alongside ref_code.
4. **KV cache prefix** (advanced) — the prefilled KV entries from the reference audio portion.
   Would enable delta-prefill-style instant reuse without re-running prefill. But KV cache
   is large (~50MB for 28 layers) and model-specific (not portable across model sizes).

**Proposed format** (`.qvoice` file):
```
magic: "QVCE" (4 bytes)
version: uint32
speaker_embedding: float[1024]      # always present
ref_text_len: uint32                 # 0 if x-vector only
ref_text: utf8[ref_text_len]         # original transcript
n_ref_frames: uint32                 # 0 if x-vector only
ref_codes: int32[n_ref_frames × 16] # codec tokens from speech encoder
```

This is compact (~4KB for x-vector only, ~20-50KB with ICL data for typical 3-10s reference)
and model-portable (same codec tokens work for both 0.6B and 1.7B Base models).

**CLI interface**:
```bash
# Create a reusable voice from reference audio
./qwen_tts -d qwen3-tts-0.6b-base --ref-audio voice.wav --ref-text "transcript" \
           --save-voice my_voice.qvoice

# Use saved voice for any text (ICL quality, no re-encoding)
./qwen_tts -d qwen3-tts-0.6b-base --load-voice my_voice.qvoice \
           --text "Any new text here" -o output.wav
```

**Tasks**:
- [x] `[MED]` Design `.qvoice` file format (speaker embedding + ref_code + ref_text)
- [x] `[MED]` Extend `--save-voice` to save full ICL data (not just x-vector)
- [x] `[MED]` Extend `--load-voice` to load `.qvoice` and reconstruct ICL prompt
- [x] `[LOW]` CLI: `--save-voice` without `--text` creates voice profile and exits
- [ ] `[LOW]` Evaluate KV cache prefix caching (is it worth the ~50MB size?)
- [x] `[LOW]` Voice library management: `--list-voices`, `--delete-voice`

---

### Phase 13: Cross-Model Voice Injection (Clone + Style Control)

**Goal**: Enable cloning any voice and using it across all model types with full style
control. One-command extraction from audio → reusable `.bin` → works on CustomVoice (with
`--instruct`), Base, and VoiceDesign models.

**Discovery**: The ECAPA-TDNN speaker embeddings (Base model) and the discrete codec speaker
embeddings (CustomVoice model) live in **compatible vector spaces** (cosine similarity ~0.94,
norms 14-17). Injecting an ECAPA embedding into CustomVoice produces the correct voice AND
supports `--instruct` for style control.

**Why this matters**:
- Preset voices (ryan, serena, etc.) sound unnatural in some languages — cloned native
  speakers sound much better
- Voice clone on Base model has no style control (instruct not supported)
- Cross-model injection gives **clone quality + instruct control** in one workflow
- RTF savings: `.bin` injection skips mel/ECAPA entirely, prefill only (fast)

**Performance comparison** (Apple M1, Italian text ~8s audio):

| Method | Prefill | Total | RTF | Notes |
|--------|---------|-------|-----|-------|
| Voice clone (Base, --ref-audio) | 5.4s | 27.9s | 3.21 | Full ECAPA extraction every time |
| Voice clone (Base, .qvoice) | 5.3s | 37.1s | 4.14 | Skip ECAPA but still Base model speed |
| Cross-model (CustomVoice, .bin) | fast | ~19.5s | 3.17 | Skip ECAPA, CustomVoice model weights |
| Cross-model + --instruct | fast | ~45s | 5.81 | Full style control on cloned voice |
| Preset speaker (CustomVoice) | fast | ~44.7s | 6.21 | Baseline for comparison |

**Status: WORKING — slight timbre shift vs direct clone, but valuable for custom voices**

Cross-model injection works: a .bin extracted from the Base model can be loaded into
CustomVoice (with or without --instruct) and produces recognizable, usable voice output.
There is a slight timbre difference vs direct voice clone on Base — the voice is similar
but not identical. This is expected because Base and CustomVoice have different transformer
weights (80-88% of attention/MLP values differ — they were trained separately).

Despite the difference, the **value is immense**: custom voices per language, reusable
voice profiles, fast generation with CustomVoice model speed (RTF ~1.5-1.7 on 0.6B).

**Key findings**:
- Only 3 of 9 preset speakers have real embeddings (ryan, vivian, serena; others ~0.02 norm)
- Auto norm scaling helps (ECAPA ~17 → CustomVoice ~14.5)
- 0.6B cross-model produces good results (closer norm match)
- 1.7B cross-model usable but more timbre shift (2 of 3 real speakers are Chinese-trained)
- Voice clone on Base model is nearly perfect — understanding WHY is key to improving cross-model

**Completed**:
- [x] `[MED]` Allow `--load-voice .bin` on CustomVoice/VoiceDesign models
- [x] `[MED]` Smarter `--instruct` warning (only on Base, not CustomVoice)
- [x] `[MED]` Better `--ref-audio` error message (suggest 2-step workflow)
- [x] `[MED]` README documentation with workflow, examples, model table
- [x] `[MED]` Auto norm scaling to match target model's preset speakers
- [x] `[MED]` Tested embedding space: cosine ~0.94, norms differ, direction blending tested
- [x] `[MED]` Verified: Base and CustomVoice transformer weights differ 80-88%

**TODO — Understanding & Analysis**:
- [ ] `[HIGH]` **Deep analysis: why does Base voice clone work so well?**
  - Trace the full pipeline: WAV → mel → ECAPA-TDNN → embedding → prefill → KV cache → generation
  - The Base model produces near-perfect clone from just 30s of audio + a 2048-dim embedding
  - How does the transformer use the embedding during attention? Is it just the initial KV entry
    at the speaker position, or does it propagate through the layers in a special way?
  - Compare what the Base transformer "does" with the embedding vs what CustomVoice does
  - This understanding is key to making cross-model portable voices
- [ ] `[HIGH]` **Analyze RTF difference: why is Base voice clone slower?**
  - Base 1.7B: RTF 3.2-4.1 vs CustomVoice 1.7B with preset: similar RTF
  - But Base 0.6B: RTF 1.5-1.8 vs CustomVoice 0.6B: RTF 1.5-1.7 (similar!)
  - Is the slowness on 1.7B Base due to RAM pressure (16GB machine, mmap paging)?
  - Or is there actual compute difference in the Base model architecture?
  - Profile: prefill time, per-frame Talker ms, per-frame CP ms — compare Base vs CustomVoice
  - If it's just RAM, a machine with 32GB would show no difference
- [ ] `[HIGH]` **KV cache dump approach**: instead of injecting raw ECAPA embedding into
  CustomVoice, dump the KV cache entries at the speaker position AFTER prefill on the
  Base model, and inject those processed KV values into CustomVoice. This would bypass
  the different-transformer-weights issue because we inject post-processed values.
  - Estimate KV cache size per position: 28 layers × 2 (K+V) × kv_dim × sizeof(bf16)
  - For 0.6B: 28 × 2 × 1024 × 2 = ~112KB per position (very small!)
  - For 1.7B: 28 × 2 × 1024 × 2 = ~112KB (same — KV uses kv_dim not hidden)
  - Could save just the speaker position's KV entries in a new format
  - Question: does CustomVoice's KV cache expect data from its own transformer or is it flexible?
- [ ] `[MED]` **Lightweight projection layer**: train a small linear projection (matrix multiply)
  that maps ECAPA embeddings → CustomVoice codec embedding space. Would need correspondences:
  generate audio of preset speakers with CustomVoice, extract ECAPA from that audio, build
  mapping from the pairs. Only 3 real speakers (ryan, vivian, serena) = very underdetermined,
  but could work with pseudo-inverse or regularized regression.

**TODO — Features & Tooling**:
- [ ] `[HIGH]` One-command voice extraction: `./qwen_tts --extract-voice ref.wav -o voice.bin`
  - Loads only speaker encoder (not full model), extracts embedding, prints usage tips
  - Detects language from audio (or accepts `--language`) and suggests matching model
  - Prints: "Voice saved. Use with: ./qwen_tts -d qwen3-tts-1.7b --load-voice voice.bin ..."
- [ ] `[HIGH]` `make extract-voice REF=speaker.wav` Makefile target for quick extraction
- [ ] `[HIGH]` Test cross-model injection on 0.6B CustomVoice — already confirmed working well
- [ ] `[HIGH]` Test cross-model with 1.7B Base .bin → 1.7B CustomVoice with/without instruct
  - Confirmed: works with slight timbre shift, still valuable for custom voices per language
- [ ] `[MED]` Test with genuinely new voices (not ryan-from-ryan, but unseen speakers)
- [ ] `[MED]` Voice gallery: pre-extract .bin files for common languages (Italian, Spanish, etc.)
  - Find native speakers for target languages, extract .bin, test quality
  - Ship as `qvoices/italian_native.bin`, `qvoices/spanish_native.bin`, etc.
- [ ] `[MED]` Evaluate quality with different norm scaling strategies
  - Raw ECAPA norms vs scaled to preset range — which sounds more faithful?
- [ ] `[LOW]` Blog post on cross-model voice injection analysis
- [ ] `[LOW]` Server API: accept .bin path in JSON request for per-request voice switching

---

### Phase 15: SDOT/SMMLA INT8 Native Dot Product (Architecture-Specific)

**Goal**: Use native int8×int8 dot product instructions (ARM SDOT / x86 VNNI) for
matvec, bypassing f32 dequantization entirely.

**Context**: Current INT8 path dequantizes to f32 before FMA (3 SIMD ops per 4 weights).
SDOT computes 4 × int8 dot products in a single instruction into int32.

| Instruction | Macro | Min Arch | Apple Silicon |
|-------------|-------|----------|---------------|
| SDOT | `__ARM_FEATURE_DOTPROD` | ARMv8.2 | M1+ |
| SMMLA | `__ARM_FEATURE_MATMUL_INT8` | ARMv8.6 | M2+ |
| VNNI | `__AVX512VNNI__` / `__AVXVNNI__` | AVX-512/AVX2 | N/A |

**Status**: Deferred — needs M2+ or AVX-512 hardware to properly test.
M1 has SDOT but not SMMLA; the 0.6B model is too small to be bandwidth-bound anyway.

- [ ] `[LOW]` Runtime feature detection (compile-time macros + runtime sysctl/getauxval)
- [ ] `[LOW]` SDOT int8 matvec kernel (int8 weights × int8 activations → int32)
- [ ] `[LOW]` Dynamic per-tensor int8 quantization of activation vector
- [ ] `[LOW]` Quality validation vs bf16 baseline

---

### Phase 16: Metal GPU / MLX

**Context**: Metal backend was previously implemented and benchmarked on M1 — **1.3x slower
than CPU NEON**. Root cause: unified memory means CPU and GPU share the same bandwidth ceiling,
and our workload (bf16 matvec) is already bandwidth-bound. GPU adds kernel launch overhead
without gaining bandwidth.

Metal 4 (2025, M3/M4) introduces tensor first-class support and ~4.7x transformer speedups.
FlashAttention on GPU could fuse QKV+attention+softmax into one kernel, reducing memory
round-trips. Higher bandwidth on newer chips (M3: 100GB/s, M4 Pro: 273GB/s vs M1: 68GB/s)
could tip the balance.

**When to revisit**: When M3/M4 hardware is available to benchmark. M1/M2 unlikely to benefit.

- [ ] `[LOW]` Prototype fused attention Metal shader (FlashAttention-style)
- [ ] `[LOW]` Benchmark vs CPU on M3/M4 (must beat CPU to justify inclusion)
- [ ] `[LOW]` Evaluate MLX C API as alternative (pre-optimized kernels, but ~50MB dependency)

---

### Phase 15: Windows Native Support

**Current**: WSL2 build works and is documented. Native Windows would need:
- mmap → MapViewOfFile wrapper
- pthreads → Windows threads or pthreads-w64
- Significant #ifdef burden

**Decision**: Only if there's real user demand. WSL2 is the recommended path.

- [ ] `[LOW]` Test full flow on WSL2 (no Windows machine currently available)
- [ ] `[LOW]` Evaluate MinGW-w64 + OpenBLAS build feasibility
- [ ] `[LOW]` Alternative: CMake + vcpkg for Windows native build

---

### Remaining Minor Tasks

- [ ] `[LOW]` CUDA/HIP backend stubs (future NVIDIA/AMD support)
- [ ] `[LOW]` Top-p partial sort with early exit (only matters when top_p < 1.0)

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
