# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-14

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

**Discovery (March 2026)**: Deep weight analysis revealed that Base and CustomVoice models
have **nearly identical transformer weights** (cosine ≈ 0.9999 per layer). The ONLY difference
is the codec_embedding table (9 preset speaker tokens) and the ECAPA-TDNN speaker encoder
(76 extra tensors in Base only). The earlier claim of "80-88% different weights" was WRONG —
that was measuring KV cache output divergence, not weight divergence.

**What this means**: Cross-model voice injection is fundamentally sound because the
transformer that processes the voice embedding is the SAME model. The fidelity gap comes
from micro-differences (cosine 0.99997 in text_projection) that accumulate through
autoregressive generation, not from a fundamental architecture mismatch.

**Why this matters**:
- Preset voices (ryan, serena, etc.) sound unnatural in some languages — cloned native
  speakers sound much better
- Voice clone on Base model has no style control (instruct not supported)
- Cross-model injection gives **clone quality + instruct control** in one workflow
- RTF savings: `.bin`/`.qvoice` injection skips mel/ECAPA entirely, prefill only (fast)

**Performance comparison** (Apple M1, Italian text ~8s audio):

| Method | Prefill | Total | RTF | Notes |
|--------|---------|-------|-----|-------|
| Voice clone (Base, --ref-audio) | 5.4s | 27.9s | 3.21 | Full ECAPA extraction every time |
| Voice clone (Base, .qvoice) | 5.3s | 37.1s | 4.14 | Skip ECAPA but still Base model speed |
| Cross-model (CustomVoice, .bin) | fast | ~19.5s | 3.17 | Skip ECAPA, CustomVoice model weights |
| Cross-model + --instruct | fast | ~45s | 5.81 | Full style control on cloned voice |
| Preset speaker (CustomVoice) | fast | ~44.7s | 6.21 | Baseline for comparison |

**Status: WORKING — slight timbre shift vs direct clone, root cause identified**

Cross-model injection works: a .qvoice extracted from the Base model can be loaded into
CustomVoice (with or without --instruct) and produces recognizable, usable voice output.
There is a slight timbre difference vs direct voice clone — the voice is similar but not
identical. Root cause analysis (March 2026) identified THREE specific sources:

1. **`tts_pad_embed` micro-difference** (cosine 0.99996 between models) — used at EVERY
   frame of generation, accumulates through 28 layers × 60+ frames
2. **Weight micro-differences** (cosine 0.9998 worst at layers 16-18 MLP) — compound
   through the attention chain
3. **Autoregressive butterfly effect** — tiny logit differences push past sampling
   thresholds → different token → cascade divergence

**Key findings**:
- Transformer weights are **identical** (cosine 0.9998-1.0000 across ALL 308 talker tensors)
- Codec embedding: codebook entries (0-2047) are **identical** between models
- Only 9 speaker preset tokens differ (norm ~0.016 in Base vs ~10 in CV)
- Base has 76 extra ECAPA-TDNN tensors (speaker_encoder.*)
- Only 3 of 9 preset speakers have real embeddings (ryan, vivian, serena; others ~0.02 norm)
- Auto norm scaling helps (ECAPA ~17 → CustomVoice ~9.4 on 0.6B)
- Same-model KV reload also diverges (mel corr 0.85) — sets upper bound on injection quality
- Cross-model injection mel correlation: 0.65 overall, 0.85 at first frames → butterfly effect

**Quantified divergence** (Silvio Italian, seed 42):

| Configuration | Mel Corr vs Base | Duration | Quality |
|--------------|------------------|----------|---------|
| Base direct (ground truth) | 1.000 | 4.48s | Perfect clone |
| Base KV reload (same model) | 0.847 | 4.32s | BF16 quantization cascade |
| CV inject (.qvoice v3) | 0.649 | 5.20s | Recognizable, slight timbre shift |

**Completed**:
- [x] `[MED]` Allow `--load-voice .bin` on CustomVoice/VoiceDesign models
- [x] `[MED]` Smarter `--instruct` warning (only on Base, not CustomVoice)
- [x] `[MED]` Better `--ref-audio` error message (suggest 2-step workflow)
- [x] `[MED]` README documentation with workflow, examples, model table
- [x] `[MED]` Auto norm scaling to match target model's preset speakers
- [x] `[MED]` Tested embedding space: cosine ~0.94, norms differ, direction blending tested
- [x] `[MED]` ~~Verified: Base and CustomVoice transformer weights differ 80-88%~~
  **CORRECTED (March 2026): weights are nearly IDENTICAL (cosine 0.9998-1.0000).
  The 80-88% claim was measuring KV output divergence, not weight divergence.**

**TODO — Understanding & Analysis**:
- [x] `[HIGH]` **Deep analysis: why does Base voice clone work so well?**
  - ANSWERED: The Base model's ECAPA embedding gets processed through 28 attention layers
    during prefill, producing rich KV entries that condition all subsequent generation.
    The KV cache carries the FULL voice identity — not just a single embedding vector.
  - PROVEN: dumping the KV and loading it cross-model preserves ~90% voice fidelity,
    far better than raw embedding injection (~60-70%)
  - KEY INSIGHT: the voice prefix (first ~10 KV positions) carries the identity.
    Separating this from the text portion enables reuse with different text + instruct.
- [x] `[HIGH]` **Deep weight analysis (March 2026)**: COMPLETE
  - ALL transformer weights are nearly identical (cosine 0.9998-1.0000)
  - Only difference: codec_embedding table (9 speaker presets) + ECAPA-TDNN (Base only)
  - Root cause of cross-model divergence identified:
    1. `tts_pad_embed` micro-difference (cosine 0.99996) → accumulates per-frame
    2. Weight micro-diffs (cosine 0.9998 at layers 16-18) → compound through layers
    3. Autoregressive butterfly effect → cascading token divergence
  - See blog/cross-model-voice-analysis.md for full analysis
- [x] `[MED]` **Analyze RTF difference: why is Base voice clone slower?**
  ANSWERED (March 2026): Base is only ~10% slower on 1.7B, negligible on 0.6B.
  Earlier measurements were contaminated by mmap cache pressure (running both models
  back-to-back on 16GB). Measured in isolation (same text, seed 42, Italian):
  - 0.6B: CV RTF 1.94, Base RTF ~2.0 (negligible difference)
  - 1.7B: CV RTF 4.58, Base RTF 5.05 (+10%)
  Root cause: Base has ~76 extra ECAPA-TDNN tensors in mmap that compete for OS page
  cache on RAM-constrained machines (16GB). Not a code issue — just mmap paging.
- [x] `[HIGH]` **KV cache dump approach**: TESTED and WORKING
  - Initial bug: kv_dim was 512 instead of 1024 (half data) → garbage output. Fixed.
  - **Same-model (Base→Base): BIT-IDENTICAL** output. Skip prefill entirely → faster.
  - **Cross-model (Base→CustomVoice): WORKS WELL** — slight timbro shift (~90% fidelity)
    but voice is clear, correct language, no artifacts. Much better than .bin embedding
    injection (~60-70%). The KV carries 28 layers of processed conditioning vs single vector.
  - Size: ~3.5MB (0.6B) to ~5MB (1.7B) for full prefill KV
  - Format: `.bin` with header (magic "QVKV", ver, n_layers, kv_dim, prefill_len) + dec_x
  - **Limitation**: current dump includes FULL prefill (voice + text). Cannot change text
    or add instruct after loading — the KV is for ONE specific prompt. Need voice-only
    prefix dump to enable text changes and instruct.
**SUPERSEDED — KV voice prefix approach abandoned in favor of WDELTA**:

The KV voice prefix idea (dump ~10 positions of voice KV, reload on target model) was
explored but WDELTA int16 with LZ4 compression proved superior in every dimension:
bit-identical output, only +7% overhead, self-contained file, works with --instruct,
no need for separate .qvkv format. The KV prefix approach would have been ~90% fidelity
at best (same as full KV dump) vs WDELTA's 100% bit-identical.

Similarly, lightweight projection layers (ECAPA→CustomVoice codec space) are unnecessary
since WDELTA already achieves perfect fidelity by applying the exact weight differences.

**Completed approaches (for reference)**:
- [x] `[MED]` .bin embedding injection: works ~60-70% fidelity, auto norm scaling
- [x] `[MED]` Full KV dump: works ~90% but tied to specific text, no instruct
- [x] `[LOW]` Blog post on cross-model voice analysis: blog/cross-model-voice-analysis.md
- [ ] `[LOW]` Server API: accept voice path in JSON for per-request voice switching

**Fidelity gap analysis — COMPLETE (March 2026)**:

Exhaustive testing revealed that **cross-model divergence ≈ same-model seed variance**.
The voice identity is preserved; only prosody varies. This is a structural limit of
autoregressive generation, not a fixable bug.

- [x] `[HIGH]` **TPAD: Save Base tts_pad/bos/eos embeds in .qvoice**: +2.3% mel corr,
  eliminates dominant per-frame drift. Cost: +12KB. Implemented.
- [x] `[HIGH]` **WOVR: Save text_projection + codec_embedding in .qvoice**: Full weight
  override for maximum cross-model fidelity. Self-contained 16MB file — no Base model
  needed to use the voice. RTF 1.60 (-20% vs clone from WAV). Implemented.
- [x] `[MED]` **.qvoice v3 with metadata**: Language auto-set, model mismatch warnings,
  voice naming. Prevents the "Italian voice used with English" mistake.
- [x] `[MED]` **Test lower temperature**: temp 0.3 WORSE (sharpens butterfly effect).
- [x] `[MED]` **Test longer reference audio**: 47s WORSE than 30s. 30s is sweet spot.
  ECAPA embeddings already very stable (cosine > 0.999 between 30s and 47s).
- [x] `[MED]` **Test greedy warmup**: WORSE (changes generation trajectory completely).
- [x] `[MED]` **Test top-k reduction**: No effect (token already in top-5 at temp 0.5).
- [x] `[MED]` **Multi-seed analysis**: Cross-model variance (0.32) = same-model variance
  (0.30). Confirmed: divergence is natural sampling noise, not a quality issue.
- [x] `[LOW]` **Verified preset voices don't interfere**: codec_embedding codebook entries
  (0-2047) are bit-identical. Speaker presets are never looked up in voice_clone mode.
- [x] `[LOW]` Per-layer KV adapter: NOT NEEDED — divergence is at seed-variance level.
- [x] `[HIGH]` **Partial layer replacement test**: Replacing top-5 or top-10 most-divergent
  layers with Base weights makes output WORSE (mel corr 0.59/0.37 vs 0.71 baseline).
  Transformer is a chain — mismatched interfaces at layer boundaries cause more harm
  than uniform micro-differences. It's all-or-nothing.
- [x] `[MED]` **Weight delta compression analysis**: 87% of BF16 values differ (mean
  |delta|=117-228 per layer). Delta gzip = 290-370 MB, not feasible for .qvoice.
  Delta gzip = 290-370 MB — too large for naive approach but promising for smarter encoding.
- [x] `[HIGH]` **WFULL: Store ALL Talker+CP weights in .qvoice**: 402 tensors, 1.7GB file.
  Produces **PCM-level BIT-IDENTICAL** output on CustomVoice. No Base model needed at runtime.
  Critical fix: CP gate_up_fused must be rebuilt after weight override (was causing codebook
  5-15 divergence). RTF 1.96. **THIS IS THE PROOF THAT PERFECT CROSS-MODEL IS POSSIBLE.**

**RTF summary (Apple M1, 0.6B, Silvio Italian)**:

| Config | .qvoice Size | RTF | Mel Corr | Fidelity |
|--------|-------------|-----|----------|----------|
| Base --ref-audio | N/A | 2.00 | ref | Perfect clone |
| Base --load-voice | 16MB | 1.78 | 1.000 | **bit-identical** |
| CV TPAD only | 16KB | 1.88 | 0.756 | Voice similar, prosody varies |
| CV WOVR | 16MB | 1.60 | 0.711 | Voice similar, prosody varies |
| **CV WFULL** | **1.7GB** | **1.96** | **1.000** | **BIT-IDENTICAL** |

**TODO — WDELTA: Compress WFULL for practical voice files**:

The 1.7GB WFULL proves bit-identical cross-model is achievable. Now compress it.
The delta distribution is very favorable for custom encoding:

```
|delta| = 0:  12.8% of values (skip — no change needed)
|delta| ≤ 1:  33.8% (1 bit + sign = 2 bits)
|delta| ≤ 3:  58.2% (2 bits + sign = 3 bits)
|delta| ≤ 7:  77.3% (3 bits + sign = 4 bits)
|delta| ≤ 15: 88.5% (4 bits + sign = 5 bits)
|delta| ≤ 127: 98.5% (7 bits + sign = 1 byte)
```

Approach: store deltas as variable-width integers with tensor index.
At load time: read CV weights from mmap, apply deltas in-place (allocate copy first).
Both models must have same architecture for deltas to apply.

- [x] `[HIGH]` **WDELTA int8: implemented, mel corr 0.82, RTF 1.49 (fastest!)**
  INT8 delta (clamp ±127), gzipped per tensor, 403 MB file. 0.9% clamp overflow
  causes CP codebooks 14-15 to diverge → not bit-identical but much better than WOVR.
  Bugs fixed: (1) F32 tensors must not be treated as BF16 delta, (2) WOVR modifies
  global pointers before WDELTA → must save/use original mmap'd pointers.
- [x] `[HIGH]` **WDELTA int16 (lossless): BIT-IDENTICAL at 510 MB!**
  Int16 deltas (no clamp), gzipped per tensor, dtype_flag=3. CONFIRMED BIT-IDENTICAL
  PCM output on both 0.6B and 1.7B. 1.7B required adding `small_to_mtp_projection`
  (weight+bias, 404 tensors total). Requires `--target-cv <cv-model-dir>` at creation.
  **PERFORMANCE ISSUE**: delta decompression overhead is significant:
  - 0.6B: ~7s (494MB gzip) → total 16.7s vs 13.8s preset (21% slower)
  - 1.7B: ~21s (1.8GB gzip) → total 54.5s vs 33.1s preset (65% slower)
  Generation RTF itself is fine (1.98 vs 2.25), the bottleneck is zlib decompress.
- [x] `[HIGH]` **Test WDELTA int16 + --instruct: WORKS!**
  Silvio voice + instruct (triste/felice/arrabbiato/solenne) on 1.7B CV confirmed.
  Voice identity preserved, styles applied. Effect is subtle — instruct modulates
  prosody more than timbre. This is expected model behavior.
- [x] `[HIGH]` **WDELTA load speedup: LZ4 compression — DONE, +7% vs preset!**
  LZ4 replaces zlib for delta compression (dtype=4). Results:
  - 0.6B: 15.9s (zlib) → 12.8s (LZ4), vs 12.0s preset = **+7% overhead only**
  - File size: 510MB (zlib) → 785MB (LZ4) = +54% larger but ~7x faster load
  - Multi-threaded decompress evaluated: NOT NEEDED — LZ4 delta load is ~1s,
    only 5% of total time. Threading complexity not worth 0.7s saving.
- [x] `[HIGH]` **WDELTA target model validation**: Stores target hidden_size in WDELTA
  header. Rejects: (1) loading on Base model, (2) size mismatch (0.6B↔1.7B).
  Existing enc_dim check provides defense-in-depth.
- [x] `[HIGH]` **Server .qvoice support**: Works via startup preload:
  `./qwen_tts -d qwen3-tts-0.6b --load-voice silvio.qvoice --serve 8080`
  Language auto-preserved from .qvoice metadata across requests. Clients just
  send `{"text":"..."}` — no need to specify language or speaker.
  Per-request voice switching not implemented (WDELTA too heavy for hot-swap).
- [x] `[MED]` **README documentation**: Update with full .qvoice workflow, WDELTA
  creation examples, LZ4 dependency note, file size/RTF comparison table.
  Document --target-cv, --voice-name flags. Add troubleshooting section.
  Server + .qvoice startup preload documented.
- [x] `[MED]` **README documentation for WDELTA voice creation**: Document the full
  workflow including requirements (need BOTH Base + CV model for creation), file naming
  conventions (e.g. `silvio_06b.qvoice` / `silvio_17b.qvoice`), metadata that encodes
  the target CV model size, and load-time validation (warn if loading 0.6B voice on 1.7B).
  Include examples:
  ```bash
  # Create voice (needs Base + CV models present for delta computation)
  ./qwen_tts -d qwen3-tts-0.6b-base --ref-audio speaker.wav -l Italian \
      --voice-name "Mario" --target-cv qwen3-tts-0.6b \
      --save-voice voices/mario_06b.qvoice
  # Use voice on any machine (only needs CV model + .qvoice file)
  ./qwen_tts -d qwen3-tts-0.6b --load-voice voices/mario_06b.qvoice \
      --text "Ciao mondo!" -o output.wav
  ```
  Must clearly explain: creation = one-time (needs both models), usage = portable
  (only CV model + .qvoice). This is the key UX story for voice cloning.

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
