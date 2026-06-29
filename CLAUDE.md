> ## ⚓ CONTINUITY PROTOCOL — DO THIS FIRST, EVERY SESSION
> Before acting, READ: **`docs/PROJECT-COMPASS.md`** (mission, how-we-work, quality criteria, the ACTIVE epic
> + experiment trail) → then the auto-memory `MEMORY.md` index → then the active plan (`plan_emo_v3.md`).
> The reasoning behind what we do — objectives, WHY, methodology, what's already decided/tried — lives there,
> not just in code. Do NOT re-derive from scratch or re-ask what was settled days ago. Keep the thread.
> When you finish a meaningful thread or experiment, UPDATE the Compass experiment-index + the relevant memory.
>
> 🎯 **EMOTION IS SOLVED — THE recipe is `docs/emotion-THE-recipe.md` (single source of truth, mirrored in
> `main.c`).** ONE rule: **preset voice → pure STEER `ryan_<emo>` @ w12** (clean, every language); **cloned voice
> → COMBINE** (language `.expr` + steer). Use the **native preset per language** (JA `ono_anna`, KO `sohee`, ZH
> `vivian`, EN/Romance `ryan`). Exposed as the `--emotion <sad|joy|anger|fear|disgust|surprise>` flag. Ear-validated
> 2026-06-29, committed + pushed. Paralinguistics = inline `[laugh]`/`[sigh]` tags. **Do NOT re-derive from the old
> abandoned methods** (graft/x-vector/τ-vectors/`.vec`/dense-FT/per-language EXPR-COMBINE — all archived in
> `docs/archive/`). Read THE recipe, don't re-spelunk weeks of experiments.

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
  - ALL actual kernels live here (inline `#ifdef __ARM_NEON / __AVX2__ / else-scalar`),
    plus common math, threading (GCD, macOS-only), BLAS paths
  - reality (verified 2026-06-03): hot matvecs/attention are NEON-or-scalar — **no AVX2**;
    only rms_norm + bf16 conversion have an AVX2 path. SDOT int8 is ARM-only. See PLAN.md 21.3.
- `qwen_tts_kernels_generic.c`, `qwen_tts_kernels_neon.c`, `qwen_tts_kernels_avx.c`
  - **EMPTY placeholder TUs** (reserved for future split). NOT where kernels live today —
    everything is inline in `qwen_tts_kernels.c`. `_avx.c` in particular has 0 AVX code.
- `qwen_tts_kernels_impl.h`
  - architecture dispatch macros (currently a stub; dispatch is via #ifdef in kernels.c)
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

## Cross-Hardware Performance Workflow (perf / SIMD / batching epics)

Dev box is **Apple M1** (the user's only fast-iteration device). M1 covers NEON + dotprod/SDOT
but **not** bf16/i8mm (M2+), SVE (server ARM), or AVX-512/VNNI/BF16/AMX (x86). So for any
performance/new-SIMD epic, the loop is:

1. **Develop on M1** here (fast iteration). Newer-ISA kernels are `#ifdef`-guarded so M1 keeps
   compiling the scalar/NEON fallback.
2. **Keep it compile-clean for other ISAs without leaving M1:** `make check-isa` forces the
   `-march` for the guarded BFMMLA/SMMLA/SME (ARM) and VNNI/BF16/AMX (x86) paths so syntax errors
   surface NOW, not on the rented box.
3. **Validate on real hardware** (we cannot fully test these on M1): rent/borrow the target, run
   the one-command harness, read the numbers, **fix, loop**. This is expected — perf/SIMD epics
   are *not* done until measured on real silicon.

**The testing/benchmark make targets** (document any new one here + in `docs/hardware-testing.md`):
- `make bench-matrix` / `make bench-matrix-full` — full per-box report: `--caps` + `--self-test`
  (native + scalar fallback) + `--matmat-bench` + RTF matrix (single/batch[/stream/server] ×
  bf16/int8/int4). Copy onto any rented ARM/x86 box. Quiet machine only.
- `make matmat-bench` — batched matmat twins (`qwen_matmat_{bf16,int8,q4_0}`) vs B×matvec, per
  precision/threads (no model needed).
- `make check-isa` — compile-check the newer-ISA kernel paths (above).
- `./qwen_tts --caps` — runtime SIMD-extension detection ("does it fire?"): ARM dotprod/bf16/i8mm/
  SVE/SME, x86 AVX-512/VNNI/BF16/AMX. `./qwen_tts --self-test` — cross-ISA kernel correctness oracle.
- Reference: **`docs/hardware-testing.md`** (which box, where to rent, the §7 per-ISA optimization
  roadmap, the fill-in RTF matrix). Keep it updated as boxes are tested.

Use **RTF + mel-corr** (not wall-clock, not md5) to compare across ISAs — fp-order differs benignly.

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

## Expressivity Plugins: Emotion vs Paralinguistics (Design Contract)

> ✅ **Emotion shipped (2026-06-29): the recipe is `docs/emotion-THE-recipe.md` — preset → STEER w12, clone →
> COMBINE, native preset per language, via `--emotion`.** The notes below are the para-vs-emo design contract
> (still relevant for the paralinguistics FT, future work); the emotion side is DONE — follow THE recipe.

Emotion and paralinguistics (`[laugh]`/`[sigh]`/`[cough]`/`[breath]`…) are **SEPARATE, COMPOSABLE plugin fine-tunes**, NOT one model:

- The emotion plugin (CSP `.expr` FT band ~16-26 **+** `--ml-steer` L21-25 steering, applied TOGETHER) is the hard-won TOP WIN for both **presets AND cloned voices**. Adding paralinguistics MUST NOT regress, overwrite, or modify it.
- Paralinguistics is an **EXTRA training-data set augmenting a DIFFERENT characteristic** that *couples* with emotion — a user who wants both loads para + emo together and quality improves, never degrades.
- Therefore the para FT must target layers **DISJOINT from the emotion plugin's layers** (few layers, CSP-style) so the two `.expr` write different `talker.model.layers.N.` tensors and compose by summation with no interference.
- Do **NOT** use a para band that overlaps emotion (e.g. 0-27 or 16-26) — it clobbers emotion. See memory `feedback_para_ft_band` + `project_para_recipe` and `plan_emo_v3.md` §9.11.

## Experiment Tracking (Senior Rule — Do Not Skip)

Every test/experiment gets its OWN dated folder + `README.md`, created with the run (not after):
`samples/tests/YYYY-MM-DD_<short-name>/` containing the audio + a `README.md` that records:
1. **Date + short name**
2. **Hypothesis** — what we want to prove
3. **Why** — the prior result that motivates it
4. **Recipe** — data, trainer hparams, layers, engine flags (exact)
5. **Eval command** — the FULL `./qwen_tts ...` line (speaker, language, instruct/emotion/steer or NONE, expr, weight, `--no-compose`, seed). Show this next to the audio — full transparency, never hide whether emotion/instruct/steering was used. Default para eval = para-expr ONLY.
6. **What to validate** — the specific ear/objective check
7. **Result + verdict** — filled after the ear verdict

Why: across a long session we run many tests; the per-test README is the durable trail of WHY/HOW/what-it-proved, so we can decide to follow or abandon a path. See memory `feedback_experiment_tracking`.

## Local-Only Artifacts (Do Not Depend On In Commits)

Common local directories/files are intentionally ignored:
- `qwen3-tts-0.6b/`, `qwen3-tts-1.7b/`
- `samples/tests/` (see below), `TODO.md`, `PLAN.md` / `plan_emo_*.md` (private notes), virtualenv folders

### ⚠️ Where audio test outputs go — NEVER loose in `samples/`
`samples/` is NOT fully gitignored: it holds a few historical sample WAVs that are LINKED from README/blog, plus
the git-tracked `samples/voice_clone_refs/**`. So a stray `git add -A` would commit GBs of audio. **Rule:**
- ALL generated audio for tests/experiments → **`samples/tests/<YYYY-MM-DD_short-name>/`** (this IS gitignored) or `/tmp`.
- NEVER write test WAVs directly under `samples/` (e.g. `samples/emotion_demo/` was wrong → use `samples/tests/emotion_demo/`).
- NEVER `git add -A` / `git add samples` — always stage files explicitly; `samples/` and `samples/voice_clone_refs/`
  contain README-linked assets you must not touch.
- The scratchpad dir (per-session, isolated) is also fine for throwaway audio.
