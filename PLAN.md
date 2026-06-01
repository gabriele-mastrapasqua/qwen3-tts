# PLAN.md — Qwen3-TTS C Engine Roadmap

Updated: 2026-03-21

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
| 0.6B BF16 | ~32 | ~79 | 1.29-1.69 | Best speed |
| 1.7B BF16 | ~80 | ~87 | 1.97-4.10 | Best quality |
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

## Phase 17: Code Cleanup & SIMD/Allocation Optimizations

**Goal**: Remove dead code, fix allocation patterns, add missing SIMD paths, improve
cache locality. Mostly quick wins with measurable impact on both ARM and x86.

### 17.1 Quick Wins (cleanup + micro-optimizations)

- [x] `[HIGH]` Remove dead `dec_up` / `cp_dec_up` NULL pointers + dead `free()` calls
- [x] `[HIGH]` Cache `codec_pad_embed` + `codec_bos_embed` at load time (avoid per-generation alloc)
- [x] `[HIGH]` Cache speaker `ref_norm` (ryan embedding norm) at load time
- [x] `[HIGH]` Align sampling work buffers (`aligned_malloc` instead of `malloc`)
- [x] `[MED]` Reduce KV cache initial alloc from 2048 → 512 (doubling handles growth)
- [x] `[MED]` Reduce RoPE cache from fixed 8192 → actual `max_tokens`
- [x] `[MED]` Prefill buffer realloc: use doubling strategy instead of exact-size

### 17.2 Scalar→SIMD (ARM NEON + x86 AVX2)

Element-wise ops in `qwen_tts_kernels.c` are pure scalar but called per-layer:

- [x] `[HIGH]` `qwen_add_inplace` — NEON + AVX2 (residual path, every layer)
- [x] `[HIGH]` `qwen_mul_inplace` — NEON + AVX2
- [x] `[HIGH]` `qwen_vec_scale_inplace` — NEON + AVX2
- [ ] `[MED]` `qwen_snake_activation` NEON — fix scalar sinf extraction (polynomial approx)

### 17.3 AVX2 Parity (x86-64 performance)

25 NEON-optimized paths but near-zero AVX2. On x86 everything falls to scalar.
Estimated ~20-30% total speedup on x86 servers/desktops:

- [ ] `[HIGH]` `qwen_rms_norm` + `qwen_rms_norm_per_head` — AVX2 FMA chains
- [ ] `[HIGH]` `qwen_causal_attention` (all 4 overloads) — AVX2 dot+accum
- [ ] `[HIGH]` `qwen_causal_attention_bf16kv` — AVX2 bf16→f32 inline dot
- [ ] `[HIGH]` `qwen_causal_attention_windowed` — AVX2
- [ ] `[MED]` `qwen_silu` — vectorized exp approximation

---

## Phase 18: Performance Push — Target RTF ≤ 1.0 (0.6B)

**Goal**: Break the RTF 1.0 barrier on 0.6B (Apple M1), push 1.7B below RTF 2.0.
Both models must benefit, all optimizations must be cross-platform (ARM + x86).

**Current baseline (Apple M1, 4 threads, long text, seed 42)**:

| Model | Talker ms/f | CP ms/f | Other | Total ms/f | RTF |
|-------|-------------|---------|-------|------------|-----|
| 0.6B  | 31.9        | 79.1    | 0.4   | 111.3      | 1.29 |
| 1.7B  | ~80         | ~87     | ~3    | ~170       | 1.97 |

**RTF 1.0 budget**: 80ms/frame (12.5 Hz). Need ~28% total reduction.

**Bottleneck analysis** (0.6B):
- Code Predictor is **67%** of per-frame time (15 sequential passes × 5 layers)
- 0.6B is at ~42% memory bandwidth utilization (28 GB/s of 68 GB/s)
- NOT bandwidth-bound → overhead and compute are the bottleneck
- 1.7B IS bandwidth-bound → INT8 helps there (already proven: ~14% CP speedup)

**Key constraint from past experiments**:
- INT8 on 0.6B Talker: 0% speedup (hidden=1024 too small)
- Speculative CP decoding: ABANDONED (codebook feedback makes it structurally unsafe)
- Metal GPU: 1.3x SLOWER on M1 (shared bandwidth ceiling)

### 18.1 CP Overhead Reduction (cross-platform)

> ⚠️ **REFUTED by micro-bench (June 2026 — see Phase 20.3).** `make cp-microbench` showed
> the CP is **90.7% matvec/GEMV** (FFN 53%, QKV+O 38%); all "overhead" ops (attention 0.7%,
> RoPE 0.1%, norms+residual ~0.5%, KV store, dispatch) total **<3%**. The tasks below target
> that <3% and are **not worth pursuing**. The CP is matvec/bandwidth-bound → the real lever
> is **CP weight quantization (18.2)** + bf16 GEMV bandwidth saturation. Kept for the record.

The CP calls `cp_transformer_step()` 17 times per frame (2 prefill + 15 decode).
Each step has 5 layers, each layer has multiple dispatch barriers and function calls.
At 42% bandwidth utilization, the gap is overhead, not raw throughput.

- [ ] `[HIGH]` **Micro-benchmark CP step**: instrument per-layer timing inside
  `cp_transformer_step` to identify exact overhead split (matvec vs norm vs attention
  vs dispatch barriers vs other). Currently we only time the whole CP.
- [ ] `[HIGH]` **Fused residual + RMSNorm**: after O-proj and down-proj, we do
  `for(i) x[i] += proj[i]` then RMSNorm reads x again. Fuse into one pass:
  `qwen_rms_norm_with_residual(out, x, residual, weight, n, eps)`.
  Saves 2 passes over x per layer × 5 layers × 17 steps = 170 cache misses/frame.
- [ ] `[MED]` **Inline CP attention for short sequences**: CP attention is at most
  pos=16 (max 17 KV entries). The generic attention kernel handles arbitrary lengths
  with SIMD; for ≤17 entries, a specialized tiny-attention kernel (fully unrolled,
  no branches) could avoid overhead. Cross-platform: just C with compiler unrolling.
- [ ] `[MED]` **Eliminate per-layer function call overhead**: flatten the 5-layer loop
  in `cp_transformer_step` into a single function with direct buffer pointers. Avoids
  layer struct indirection and enables the compiler to keep registers live across layers.
- [ ] `[LOW]` **RoPE precompute for CP positions 0-16**: CP only uses 17 positions.
  Store cos/sin as a flat [17 × head_dim] array looked up by index — cheaper than
  the generic RoPE function that handles arbitrary positions.

### 18.2 INT8 Code Predictor (cross-platform, BOTH models)

> ✅ **CORRECTED via safetensors tensor shapes (June 2026).** The Code Predictor is
> **structurally IDENTICAL on 0.6B and 1.7B**: hidden=1024, inter=3072, q_dim=2048
> (num_heads 16 × head_dim 128), 5 layers. Verified from `talker.code_predictor.model.*`
> tensor shapes (`gate_proj [3072,1024]`, `input_layernorm [1024]`) in BOTH model files.
> Only the **Talker** differs (0.6B hidden=1024 / 1.7B hidden=2048). The earlier belief that
> "1.7B CP hidden=2048" was WRONG — it conflated the CP q_dim (2048) or the 1.7B Talker
> hidden (2048) with the CP hidden (1024).
>
> Consequences: (a) the gate `cp_h >= 2048` blocks CP INT8 on **both** models (`--int8` only
> quantizes the Talker, never the CP — confirmed: 1.7B `--int8` prints "Quantizing Talker"
> but no CP). (b) There is no "easy 1.7B CP win" — the 1.7B CP is the same 1024-hidden
> matrices as 0.6B and would hit the same hang. (c) Fixing INT8 CP at hidden=1024 benefits
> **both** models equally (CP ≈ 77–87 ms/f, 90% matvec, the bottleneck on both).

> 🛑 **ROOT CAUSE of the "hang" found (June 2026): DENORMAL floats.** INT8 dequant drives
> activations into the subnormal range; denormal FP arithmetic is ~100x slower (looks like a
> hang, not a runaway — the `-m` cap works, each frame just takes seconds). `-ffast-math` does
> NOT guarantee flush-to-zero (FTZ) at runtime. Proven: enabling FTZ (FPCR bit 24 on ARM) made
> single-thread (`-j1`) int8 **complete** instead of hang. Also explains why `--int8` 1.7B was
> never really usable — combined with the old "silently skipped" bug, the "+14%/recommended"
> claim was never validated against a real int8 run.
>
> **STILL OPEN (multi-thread):** with FTZ added to all GCD worker matvec blocks + main thread,
> `-j1` works but the **default 4-thread run still hangs**. `sample` pins it at 100% in
> `qwen_matvec_int8_qkv`'s inline GCD block (`__qwen_matvec_int8_qkv_block_invoke`) during
> `qwen_talker_step` — even though FTZ is set at the top of that block. The fused-path (`-j1`)
> int8 qkv works but the threaded **inline NEON qkv** path does not. Hypotheses to try next:
> (1) route int8 qkv through the fused path always (drop the inline block); (2) check whether
> FZ truly flushes denormal *inputs* in that NEON FMA sequence on Apple Silicon (may need to
> pre-flush `x`, or the denormals are produced upstream and the real fix is numerical, not FTZ).
> NOTE: prefill uses `cblas_sgemm` (Accelerate, own threads) — not the cause here (hang is in
> the per-token step, post-prefill).

INT8 for CP specifically. The CP is greedy (temp=0, argmax) so quantization has zero
quality impact on output. The Talker INT8 was 0% on 0.6B but the CP has different
characteristics (15+ sequential calls/frame → many weight re-reads, so bandwidth-bound).

- [ ] `[HIGH]` **Make INT8 CP work at hidden=1024 (both models)**: the only real task. The
  forced-on attempt hangs (never reaches EOS) — diagnose precision vs bug, try per-channel
  scales. Expected CP speedup if solved: the CP is ~90% matvec/bandwidth → meaningful.
- [ ] `[MED]` **Re-test INT8 CP on 0.6B**: The old "0% speedup" test was for the Talker.
  CP has 15× more weight reads per frame — might cross the bandwidth threshold.
  Must measure, not assume.
- [ ] `[MED]` **INT8 argmax_matvec for CP lm_heads**: 15 lm_head matvecs per frame
  (2048→2048 each). Quantizing these to INT8 halves their bandwidth.

### 18.3 Sliding Window Attention in CP

The model config has `sliding_window=72` but it's unclear if we enforce it.
For long sequences (200+ frames), the CP KV cache grows and attention becomes O(N).
With a window, attention cost stays constant.

- [ ] `[MED]` **Verify CP sliding window**: check if CP attention already caps at 72.
  If not, implement windowed attention for CP. For short sequences (<72) no change;
  for long sequences prevents linear attention growth.

### 18.4 AVX2 Kernel Parity (x86 performance)

Currently 25+ NEON-optimized paths but near-zero AVX2. On x86 servers/desktops,
everything falls back to scalar — estimated 20-30% total speedup from AVX2 parity.
Cross-platform by definition (compile-time dispatch).

- [ ] `[HIGH]` `qwen_rms_norm` — AVX2 FMA sum-of-squares + scaling
- [ ] `[HIGH]` `qwen_causal_attention_bf16kv` — AVX2 bf16→f32 dot product + accum
- [ ] `[MED]` `qwen_causal_attention` — AVX2 (f32 KV path)
- [ ] `[MED]` `qwen_causal_attention_windowed` — AVX2
- [ ] `[MED]` `qwen_rms_norm_per_head` — AVX2

### 18.5 Prefetch and Memory Access (cross-platform)

At 42% bandwidth utilization on 0.6B, we're leaving performance on the table.
Software prefetch hints can help the memory controller prepare cache lines.

- [ ] `[MED]` **Weight prefetch in matvec**: while processing row N, prefetch row N+2.
  ARM: `__builtin_prefetch(addr, 0, 1)`. x86: `_mm_prefetch(addr, _MM_HINT_T1)`.
  Must be measured — prefetch can hurt if the hardware prefetcher is already doing well.
- [ ] `[LOW]` **madvise(MADV_SEQUENTIAL)**: hint the OS that weight reads are sequential.
  Already implicit with mmap but explicit might help on some kernels.

### 18.6 Benchmark and Validation

- [x] `[HIGH]` **Per-component CP micro-benchmark**: DONE via `make cp-microbench` (compile
  flag `-DCP_MICROBENCH`, separate `qwen_tts_cpbench` binary, zero overhead on normal build).
  Partitions CP ms/f among QKV/attn/FFN/norm/lm_head/embed. Result in 20.3. (A generalized
  `--bench` flag for the whole pipeline is still optional — the existing summary already
  reports Talker/CP/embed/codec-head per-frame.)
- [ ] `[HIGH]` **A/B test each optimization**: before/after RTF with identical params
  (seed 42, ryan, English long text). Both 0.6B and 1.7B. ⚠️ **Validate by RTF (timing) +
  perceptual/mel-corr, NOT md5** — see determinism note below.
- [ ] `[MED]` **x86 benchmark**: test on a Linux x86 machine to validate AVX2 gains.

> ℹ️ **Non-determinism — fully investigated, turns out BENIGN (June 2026).** Output is not
> always bit-reproducible run-to-run, but the difference is **±1 LSB of 16-bit PCM (~3e-5,
> −90 dB, corr = 1.0000000)** — pure floating-point rounding, **NOT an audible artifact and
> NOT a correctness bug**. Measured, not assumed. Disproven hypotheses: uninitialized read
> (`MallocScribble`/`MallocPreScribble` didn't change it), and "decoder seam artifacts" (the
> cross-chunk-size difference is ±1 LSB, corr 1.0 — `conv_rf=20` is perceptually identical to
> `conv_rf=64`).
>
> **Two benign contributors:**
> 1. **Sampling at temp>0 + parallel-matvec FP noise.** Threaded matvec sums in
>    non-deterministic order → logits differ at FP level → at temp 0.9 the softmax sample
>    occasionally flips a token. **Vanishes at `--temperature 0`** (greedy argmax is robust).
> 2. **Decoder chunk-boundary timing.** The overlapped decoder (`decoder_thread_fn`, always
>    on) consumes a variable frame count per chunk (`avail`, timing-dependent); the streaming
>    decoder isn't bit-exactly chunk-invariant, so different boundaries shift the last bit.
>    **Intermittent / load-dependent** — on a quiet machine, timing repeats and runs are
>    bit-identical (verified: long text, 4 default runs → identical `f5ea31f6`).
>
> **Receptive field, for reference:** the conv decoder's true left-context RF is ~55–64
> latent frames (measured by sweeping `conv_rf` to bit-match a full decode); the shipped
> `QWEN_SD_STREAM_CONV_RF=20` is below that, but the resulting difference is only ±1 LSB, so
> it has never mattered perceptually.
>
> **Decision: Option A (chunk-invariant decoder) NOT pursued** — it would add per-chunk
> recompute overhead to "fix" a −90 dB difference with zero audible benefit. The decoder is
> fine. For bit-stable A/B regression, run `-j 1 --temperature 0` (deterministic in practice)
> or compare by RTF + mel-corr / RMS rather than md5.

### Experiments That Didn't Work (Phase 18)
- **pthread thread pool (replace GCD)**: 8% SLOWER on macOS. Apple's GCD is
  kernel-optimized with priority inheritance; pthreads mutex+cond can't compete.
  Would help on Linux (no GCD), not on macOS. The "30% dispatch overhead" in profiling
  is intrinsic synchronization cost, not fixable with a different pool.
- **4-row fused matvec (16 elem/iter)**: 7% SLOWER than 2-row (32 elem/iter). 16 NEON
  accumulators cause register spill; shorter inner loop can't hide memory latency.
- **Threading threshold 2048**: 3% SLOWER. Even 1024-row matrices (O-proj, down) benefit
  from 4-thread dispatch on M1 despite the GCD overhead.
- **INT8 on 0.6B CP**: Generation hangs (never reaches EOS). INT8 quantization of
  hidden=1024 matrices causes numerical issues. Auto-skipped when hidden < 2048.
- **INT8 load bug (fixed)**: `--int8` flag was set after model load — quantization was
  silently skipped on ALL models since first implementation. Fixed in v0.8.0.

### Measured Impact

| Optimization | 0.6B speedup | 1.7B speedup | Status |
|---|---|---|---|
| Fused residual+RMSNorm (18.1) | **-21% short** | stable | **DONE** |
| Software prefetch (18.5) | ~1-2% | ~1-2% | DONE |
| pthread thread pool (18.1) | -8% (worse) | untested | ABANDONED |
| 4-row matvec (18.1) | -7% (worse) | untested | ABANDONED |
| INT8 CP 0.6B (18.2) | hang | N/A | ABANDONED |

### Remaining Opportunities

| Optimization | 0.6B speedup | 1.7B speedup | Effort |
|---|---|---|---|
| ~~Pipeline CP↔Talker overlap~~ | ~~10-15%~~ | ~~10-15%~~ | **INFEASIBLE** |
| INT8 CP (18.2, both models, hidden=1024) | TBD | TBD | needs hang fix |
| Sliding window CP (18.3) | 5% (long) | 5% | 0.5 days |
| AVX2 parity (18.4) | N/A (ARM) | N/A (ARM) | 2-3 days |
| **Combined** | **~25-35%** | **~30-45%** | ~1-2 weeks |

**Target RTF after Phase 18**:
- 0.6B: 1.29 → **0.85-0.95** (long text), **1.0-1.2** (short text)
- 1.7B: 1.97 → **1.2-1.5** (long text)

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

### Phase 19: Long-Form Generation Test (Audiobook-style)

**Goal**: Stress-test the engine on long-running synthesis with a custom voice, e.g. one
full book chapter (Italian) — a realistic audiobook use case. Doubles as a long-running
soak test for memory leaks, RTF drift, KV cache growth, and thermal throttling.

**Why**: Current tests are short phrases. We have no signal on:
- Memory leaks / fragmentation over thousands of frames
- RTF degradation across a multi-minute run
- Stability of KV cache and sampler state over many consecutive utterances
- UX for a run that will realistically take many minutes

**Ideas to explore**:
- [ ] `[MED]` Batch/chapter mode: input is a text file (or chapter split into paragraphs),
  output is one long WAV (or per-paragraph WAVs + concat)
- [ ] `[MED]` CLI progress indicator: `[chunk i/N] elapsed Xs, ETA Ys, RTF=Z` on stderr
  (respect `--silent`); include running avg RTF and per-chunk RTF to spot drift
- [ ] `[MED]` Long-run telemetry: periodic RSS dump, RTF per chunk, optional CSV log
  for post-mortem plots (detect leaks / thermal throttling)
- [ ] `[LOW]` Resumability: checkpoint at paragraph boundary so a crash/cancel doesn't
  lose the whole chapter
- [ ] `[LOW]` Reference target: 1 Italian chapter (~N minutes of audio) with custom voice
  (e.g. Silvio via .qvkv) as the canonical long-form test input
- [ ] `[LOW]` Add `make test-longform` target running the chapter test and asserting
  no RTF regression vs a baseline

#### Lightweight markup format (no XML, zero deps)

Professional audiobooks use SSML (XML) or hand-annotated scripts. SSML is the industry
standard but needs an XML parser — overkill for our use. Instead, define a **minimal
bracket-based markup** parseable with a tiny hand-written tokenizer in C:

```
[pause:500]          → insert 500 ms of silence in the WAV
[pause:short]        → aliases: short=300ms, medium=700ms, long=1500ms, scene=2500ms
[emph]parola[/emph]  → wrap chunk, hint the model via instruct ("emphasize 'parola'")
[voice:silvio]       → switch active custom voice (.qvkv) until next [voice:...]
[instruct: slow, sad tone] ...text... [/instruct]   → per-chunk instruct prompt (1.7B only)
[pron:schedule=ˈskɛdjuːl]   → pronunciation override (future, if needed)
# comment line       → ignored, for human annotations in the script
\n\n                 → paragraph boundary → auto [pause:medium]
.  !  ?              → sentence boundary → auto [pause:short]
```

Rules:
- Chunks are split on markup boundaries; each chunk is a separate synthesis call
- Silences are raw PCM zeros spliced into the output WAV (no model cost)
- Unknown tags are logged and stripped (forward-compatible)
- Plain txt works as-is: auto-pauses on `.`/`?`/`!`/`\n\n`, no tags needed

- [ ] `[MED]` Define the grammar formally (single-pass scanner, no regex, no XML)
- [ ] `[MED]` Implement parser + chunk scheduler in a new `qwen_tts_script.c`
- [ ] `[LOW]` Produce a realistic sample script: take one public-domain Italian chapter
  (e.g. Pinocchio cap. 1, Pirandello, or Promessi Sposi) and hand-annotate it with
  pauses/emphasis/voice switches as a reference fixture under `samples/longform/`
- [ ] `[LOW]` Ship 2 fixtures: `chapter_plain.txt` (no markup) and `chapter_annotated.qts`
  (same text, fully annotated) — diff them to showcase the format

Example annotated fragment (invented but realistic, Pinocchio-style):

```
# Capitolo 1 — Come andò che Mastro Ciliegia...
[voice:silvio]
[instruct: narrator, warm, slightly amused tone]
C'era una volta...[pause:short] —[pause:200] Un re! —[pause:medium]
diranno subito i miei piccoli lettori.[pause:long]

No, ragazzi, avete [emph]sbagliato[/emph]. C'era una volta un [emph]pezzo di legno[/emph].
[pause:scene]

Non era un legno di lusso, ma un semplice pezzo da catasta...
[/instruct]
```

---

## Phase 20: Ideas from `faster-qwen3-tts` (Andres Marafioti) + CPU/build safety

Analysis of [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (talk:
"Reachy Mini", AI Engineer). His 3 tricks and what is/isn't transferable to our pure-C
CPU engine.

### 20.0 Key insight — why his 5.8× does NOT transfer

His own slide says it: **"Compute-bound? No. Launch-overhead-bound."** On GPU the talker
(28 layers) + code predictor (5 layers × 15 tokens) issue **~500 tiny CUDA kernel
launches per decode step**, and the GPU idles between each one waiting for Python to
dispatch the next. His entire 5.8× comes from **removing that overhead** via CUDA graph
capture.

A pure-C engine **does not pay that tax**: no Python dispatch, no kernel-launch overhead,
direct inline calls. So:
- Trick #1 (static KV cache): **we already do this** — KV is pre-allocated
  `[num_layers × kv_max × kv_dim]`, scratch buffers (`codes`, `audio_buf`, `emb_cache`,
  `emb_tmp*`) allocated once, not per-frame.
- Trick #2 (CUDA graph capture): **N/A** — cures a problem (Python/launch overhead) we
  don't have. Our RTF 1.3–1.7 already reflects being compute-bound.
- Trick #3 (streaming): **the one genuinely transferable idea** — it's perceived-latency
  UX, not throughput.

> Bottom line: most of his speedup is overhead removal we already get for free. Our real
> lever (proven by the CP micro-bench in 20.3) is **CP weight quantization on BOTH models** —
> the CP is 91% matvec/bandwidth-bound, even on 0.6B (the old "0.6B not bandwidth-bound" was
> the Talker, not the CP). NOT graph capture, NOT the per-op micro-opts of 18.1 (<3% of CP),
> NOT sampling (0.35 ms/f). Streaming remains the separate latency (TTFA) lever.

### 20.1 Streaming chunked decode — refine existing infra `[MED]`

We already have streaming scaffolding (`qwen_sd_stream_state_t`, `qwen_sd_stream_free`,
`decoder_thread_t`, `--stream`). Adopt/verify his concrete params:
- [ ] **25-frame left-context sliding window** to avoid chunk-boundary artifacts.
- [ ] `chunk_size=8 → ~667 ms audio/chunk` ratio as reference.
- [ ] Measure **real TTFA on CPU** with vs without streaming (M1, 4 threads). On CPU,
  TTFA is higher than GPU so streaming matters *more*: turns "wait for whole utterance"
  into "hear audio after first chunk". No compute saved, big perceived-latency win.

### 20.2 Short prefill (x-vector-only voice cloning) — DOES NOT apply to our HQ formats `[LOW / mostly N/A]`

His x-vector mode: **10-token prefill vs ~80+ in full ICL clone**. Re-checked against our
actual voice formats (June 2026) — **the idea does not transfer to our flagship format**:
- Our shipping HQ format is **WDELTA `.qvoice` (int16+LZ4, bit-identical)**: voice identity
  comes from **weight deltas applied at load time**, NOT from a voice prefill. The prompt is
  plain text → **TTFA equals a preset voice**. The cost is **LZ4 delta decompression at load
  (~1s, +7%)**, a completely different bottleneck than prefill length.
- `.qvkv` (KV prefix, the format the original note referenced) is **ABANDONED** — superseded
  by WDELTA (see Phase 13).
- Short prefill (x-vector-only) only applies to the **Base-ICL path**, where it's exactly
  the quality↔speed trade-off we **already characterized**: x-vector-only = our `.bin`
  (~60-70% fidelity), a quality downgrade. ICL (ref_codes+ref_text) is the HQ path.
- [ ] (optional, low value) Only revisit if a latency-over-fidelity Base-ICL mode is ever
  wanted; otherwise nothing to do — TTFA on HQ voices is already preset-level.

> Correction: prefill length is **not** a lever for our HQ voices. The real latency lever
> is streaming (20.1), and for WDELTA specifically, faster delta decompression at load.

### 20.3 Our real CPU levers — grounded in OUR measured data (NOT generic) `[HIGH value]`

Corrected June 2026 after re-reading our quantization tests (`docs/quantization.md`,
Phase 18). The earlier "quantization is candidate #1" framing was **wrong for the 0.6B**.
The honest, per-model picture:

**Quantization reality (measured, do not re-litigate):**
- INT8: **~14-15% speedup on 1.7B ONLY**. **0% on 0.6B** (hidden=1024 too small).
- INT4 Q4_0: **4% SLOWER even on 1.7B** (nibble unpack > bandwidth saving). Opt-in only.
- INT8 on 0.6B CP: **hangs** (numerical issues at hidden=1024) → auto-skipped < hidden 2048.
- Root cause: **0.6B runs at only ~42% memory-bandwidth utilization → NOT bandwidth-bound.**
  Nothing to free by quantizing. Only the 1.7B is bandwidth-bound.

**CP micro-bench (June 2026, `make cp-microbench`, 0.6B, Italian, seed 42, 114 frames):**
The instrumented TOTAL (78.49 ms/f) matches the externally-measured CP total — gap-free.

| CP sub-op | ms/f | % of CP |
|---|---|---|
| FFN gate_up (matvec) | 26.86 | **34.2%** |
| QKV proj (matvec) | 19.25 | 24.5% |
| FFN down (matvec) | 14.67 | 18.7% |
| O proj (matvec) | 10.46 | 13.3% |
| lm_head argmax (matvec) | 3.37 | 4.3% |
| Embed+project | 2.55 | 3.2% |
| Attention (compute) | 0.54 | 0.7% |
| SwiGLU | 0.38 | 0.5% |
| Resid+Norm + Q/K norm + RoPE + KV store + input norm | ~0.41 | ~0.5% |

**→ matvec/GEMV = 90.7% of CP. FFN alone (gate_up+down) = 53%. All "overhead" ops
(attention, RoPE, norm, KV store, dispatch) = <3% combined.**

**This REFUTES the Phase 18.1 "CP is overhead-bound" hypothesis.** The CP is
**matvec/bandwidth-bound**, even on 0.6B. Mechanism: CP weights (~120 MB) don't fit in
cache (M1 L2 ~12 MB); the 15 codebooks are sequentially dependent (can't batch — why
speculative CP was abandoned), so the CP runs **16 serial steps/frame, each re-reading all
5 layers' weights from DRAM** → ~24 GB/s of 68 GB/s peak = the PLAN's "42% bandwidth". The
old "INT8 0.6B = 0%" was measured on the **Talker** (not bandwidth-bound) — the CP differs.
`matvec_bf16` is a hand-rolled NEON kernel (`bf16_matvec_fused`), NOT BLAS.

Corrected levers:
- [ ] **Both models — CP weight quantization is THE lever** (Phase 18.2). Micro-bench proves
  91% of CP is weight reads → cutting bytes/weight attacks it directly. The CP is **identical
  on both models (hidden=1024)** — verified via tensor shapes — so this is ONE task, not a
  "1.7B-only win". Blocker: INT8-CP hangs at hidden=1024 (both models would) → retry with
  **per-channel scales**. INT4 (which LOST on the bandwidth-unbound Talker) may finally WIN on
  the bandwidth-bound CP — measure, don't assume.
- [ ] **Secondary — bf16 GEMV bandwidth saturation**: CP matvec sits at ~42% of peak DRAM
  bandwidth → investigate dispatch granularity (small matrices called 80×/frame) + prefetch
  to push higher. Lower ceiling than quantization but stacks with it.
- [ ] **DO NOT pursue** the 18.1 micro-opts (inline tiny-attention, RoPE precompute, flatten
  loop, fuse norms): measured to target <3% of CP. Fused residual+RMSNorm already landed.
- [ ] **Both**: `make cp-microbench` gives the per-op CP breakdown to A/B each change.
- [x] Sampling: **CONFIRMED negligible** — measured June 2026 on M1, 0.6B, Italian:
  Codec head+sampling = 40 ms over 114 frames = **0.35 ms/f**. The old "sampling is
  bottleneck" note is stale (predates quickselect). Do NOT optimize sampling.

**Measured baseline (June 2026, M1, 0.6B, Italian, seed 42, ryan, 114 frames / 9.1s):**

| Component | ms/f | Share of loop |
|---|---|---|
| Code Predictor | 86.6 | **74%** |
| Talker step | 30.6 | 26% |
| Embed | ~0.15 | ~0% |
| Codec head + sampling | 0.35 | ~0% |
| **Critical-path loop** | **~117.7** | RTF loop ≈ 1.47 |

Prefill 1650 ms (51 tokens, one-time → all TTFA). Speech decoder overlapped (only 877 ms
drain additive). Whole run: 9.1s audio in 16.0s → **RTF 1.76**. Confirms: CP is the
bottleneck (even higher than the old 67% estimate); sampling/decoder are not.

> Net: CP weight quantization (18.2) is THE lever for BOTH models — the CP is identical
> (hidden=1024) on 0.6B and 1.7B. The blocker is making INT8 work at hidden=1024.
> Sampling and decoder are confirmed non-bottlenecks. Prefill (1.65s) is pure TTFA →
> streaming (20.1) is the latency lever.

### 20.4 Makefile / binary CPU-safety — latent bug we may have missed `[HIGH]`

`CFLAGS_BASE` uses **`-march=native`**. A release binary built on an AVX512/AVX2 machine
will **SIGILL** on a CPU lacking those instructions. This bites the planned release
binaries (Phase 9 CI/CD).
- [ ] Add **runtime CPU feature detection (cpuid) + kernel dispatch** instead of relying
  on compile-time `-march=native`. Today kernels are compile-time `#ifdef` inline in
  `qwen_tts_kernels.c`; `*_avx.c`/`*_neon.c` are still empty/reserved.
- [ ] For release builds: baseline `-mavx2 -mfma` (or runtime multiversioning) so binaries
  run on any x86-64-v3 CPU; keep `-march=native` only for local/dev builds.
- [ ] This runtime-dispatch work is the **prerequisite** for any future AVX512 path.

### 20.5 AVX512 — feasibility & how to verify without hardware `[LOW / deferred]`

User question: a user asked about AVX512 support; dev hardware is **Apple M1 (ARM)**.
- **Blocker**: AVX512 is x86-only — can't run natively on M1.
- **Intel SDE** (correct ISA-verification tool, `-chip-check` flags illegal instructions
  on a target chip) runs **only on x86 hosts** → unusable on the M1 locally.
- **QEMU user-mode on M1**: lists AVX512 features but AVX on aarch64 hosts is **known
  buggy/unreliable** (falsely reports AVX available) → not trustworthy for verification.
- **Realistic path**: x86 cloud box (spot Ice Lake / Sapphire Rapids) or **GitHub Actions
  x86 runner** + **Intel SDE** for ISA correctness. Belongs in CI (Phase 9), NOT local.
- **Is it worth implementing blind?** ROI is low:
  1. Heavy GEMMs already go through **BLAS (Accelerate/OpenBLAS)**, which **already
     dispatches to AVX512 at runtime** if the CPU supports it — big compute already covered
     with zero code.
  2. Our hand kernels (RoPE, rms_norm, snake, bf16 dequant, attention) are memory-bound;
     AVX512 vs AVX2 gives modest gains, and AVX512 can **downclock** on older Intel.
  3. Can verify *correctness* (SDE/CI) but **cannot tune performance without real HW**.
- [ ] Decision: **defer AVX512 hand-kernels.** First do 20.4 (runtime dispatch) + ensure
  x86 links an AVX512-capable BLAS. Only add `#ifdef __AVX512F__` paths (with fallback,
  SDE-verified in CI, marked "unverified perf") if profiling on real HW shows the auxiliary
  kernels are hot.

### 20.6 Prerequisite — re-download models `[BLOCKER]`

Local model dirs were **removed** and must be re-downloaded before any of the above can be
tested/benchmarked:
- [ ] `./download_model.sh --model small`  → `qwen3-tts-0.6b/`
- [ ] `./download_model.sh --model large`  → `qwen3-tts-1.7b/` (for 1.7B / INT8 work)
- [ ] base variants if voice-clone prefill work (20.2) is tackled:
  `--model base-small` / `--model base-large`

---

## References

- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Report (arXiv:2601.15621)](https://arxiv.org/abs/2601.15621)
- [HuggingFace: 0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
- [HuggingFace: 1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Community: Qwen3-TTS_server (FastAPI)](https://github.com/ValyrianTech/Qwen3-TTS_server)
- [Community: OpenAI-compatible API](https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi)
- [Community: Streaming implementation](https://github.com/rekuenkdr/Qwen3-TTS-streaming)
