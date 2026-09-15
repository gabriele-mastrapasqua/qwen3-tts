# CUDA parity track — 2026-09-15

Scope: get the CUDA path to a defensible state BEFORE renting a GPU box, so instance time is
spent on verification and fixes, not on discovery. Order fixed by the owner: **fixes first,
then the parity analysis, then any CUDA-only feature flags.**

Everything below marked STATIC was established on the dev Mac by reading the code or by a CPU
experiment. Nothing here is a GPU measurement; items marked NEEDS-GPU are the shopping list
for the rented box.

## 1. Fixes landed before the box

### 1.1 REPRO-1 — cross-request delta-prefill leak (FIXED, CPU-verified)

`qwen_tts_generate()` keeps `prev_input_embeds` and re-prefills only from the first position
whose embedding differs (`qwen_tts.c:1515-1546`), reusing the KV rows of the common prefix.
Causally sound, but NOT bit-identical: the tail is then computed in a shorter prefill with
different GEMM tiling and accumulation order, and over ~96 autoregressive frames the
difference forks the trajectory.

Evidence chain (0.6B, `-j1`, temperature 0, seed 42, ryan/English):

| arm | r1 | r2 | r3 | vs CLI oracle |
| --- | --- | --- | --- | --- |
| simple server, before | `4a0cd0f3` | `9ea67803` | `9ea67803` | r1 0.92744, r2/r3 **1.00000** |
| `QWEN_SD_INT8=0` | identical to above | | | flags provably applied, output byte-identical |
| `QWEN_SD_RES1_V2=0` | identical to above | | | ditto |
| A then B then B | B#1 `4a0cd0f3` | B#2 `9ea67803` | | **B-after-different-text is the wrong one** |
| `QWEN_NO_PREWARM=1`, A then B | B `4a0cd0f3` | | | prewarm is NOT the cause |
| batched server `--batch-size 2` | `fd917367` | `fd917367` | | **1.00000 — never affected** |
| simple server, after fix | `9ea67803` | `9ea67803` | `9ea67803` | **1.00000** |

Readings that matter:

* The CLI is the reference path and it matches r2/r3, so **request 1 was the anomaly** —
  `test-serve-repro` took r1 as its baseline and reported the two correct requests as failures.
* The rule is not "the first request": it is **any request whose text differs from the
  immediately preceding one**. An exact repeat matches the whole prefix, trips
  `if (delta_start >= prefill_len) delta_start = 0`, recomputes in full and is therefore right.
* **The batched server was never affected**, because `qwen_tts_generate_batch()` already clears
  `prev_prefill_len` per item (`qwen_tts.c:2091`), as do `generate_batch_multi` (`:2276`) and
  compose (`qwen_tts_compose.c:291`). Production serving (batched + prefork) and every
  qualified operating point from the Arm/Axion/Turin campaigns are unaffected. The defect was
  confined to the single-process `--serve` path, which is the default when `--batch-size` is
  not passed.
* The lazily built quantized weight packs are REFUTED: `QWEN_SD_INT8=0` and
  `QWEN_SD_RES1_V2=0` leave the output byte-identical, so they do not participate.

Fix: `reset_request_state()` (`qwen_tts_server.c:908`) now clears `ctx->prev_prefill_len`,
making the single-process server agree with the batched server and with the CLI. Server-scoped;
the CLI and the engine are untouched. `make test-serve-repro` goes from
`ndiff=118333 (81.094%)` to `ndiff=0 (0.000%)`.

Note for the GPU box: the same delta-prefill is what issue #19 fixed for the fused GPU path,
and the guard at `qwen_tts.c:1529-1543` still only forces `delta_start = 0` for a fused owner
**with steering active**. NEEDS-GPU: check whether a fused CUDA talker without steering has the
same cross-request fork.

### 1.2 TQ-7 — `--backend <gpu> --prefork N` (GUARD LANDED, NEEDS-GPU to verify)

A GPU context does not survive `fork()`. `qwen_backend_init()` runs at `main.c:1652` and the
resident CUDA Talker/CP state at `:1661-1678`, both in the parent; `qwen_tts_serve_prefork()`
forks at `:3082`. The child re-initialises the thread pool and the cost map
(`qwen_tts_server.c:2910-2911`) but there is no `qwen_backend_after_fork()`, so workers
inherited a dead handle while the banner still advertised offload. `main.c` now refuses the
combination, inside the GPU `#if` and only when `--backend` was passed, so a CPU-only build
does not compile it at all.

### 1.3 Misleading offload banner (FIXED, STATIC)

The seam in `qwen_tts_backend.h` carries `matvec_bf16` and `matmat_bf16` and nothing else, so
`--backend cuda --int8` (or `--int4`) offloads **nothing** while the startup line claims it
does. `main.c` now prints an explicit NOTE in that case and names the resident paths instead.

## 2. Parity gaps found statically — the GPU shopping list

| # | Gap | Where | Status |
| --- | --- | --- | --- |
| P1 | Packed ConvTranspose layout: `sd_pack_convt` writes `[k][ic][oc]` and overwrites the weight pointer in place (`qwen_tts_speech_decoder.c:229`, `:1321`, `:1338`), so CUDA receives the packed tensor while `kd_convT` read `[ic][oc][k]` | `qwen_tts_cuda_decoder.cu` | **PR #29 fixes it**; diagnosis confirmed statically |
| P2 | Backend seam is bf16-only, so no quantized path has a GPU counterpart | `qwen_tts_backend.h` | by design; banner now honest |
| P3 | `QWEN_CUDA_CONVDEC=1` disables the exact streaming decoder (`sd_exact_stream_enabled()` returns 0) | `qwen_tts_speech_decoder.c:3062` | NEEDS-GPU: quantify what streaming loses |
| P4 | `QWEN_CUDA_CONVDEC` + non-streaming forces `dt_no_overlap = 1`, dropping decoder/talker overlap | `qwen_tts.c:1696` | NEEDS-GPU |
| P5 | Batched CUDA requires fused talker AND CP and is capped at `B <= 8` | `qwen_tts.c:2964` | NEEDS-GPU: is the cap real or arbitrary? |
| P6 | Prefork + GPU unsupported, so GPU serving scales only via `--batch-size` in one process | `main.c` guard | design question, not a bug |

Positive: every CUDA env flag (`QWEN_CUDA_FUSED_TALKER`, `QWEN_CUDA_DECODER`, `QWEN_CUDA_CONVDEC`,
`QWEN_CUDA_BATCH`, `QWEN_DEC_NAIVET`) is already declared in `g_qwen_reported_flags[]`, so
`make check-flag-registry` and the `[FLAGS]` line cover them.

## 3. Which v2 ideas can cross to CUDA

The v2 work is CPU-topology shaped (core slicing, per-worker masks, NUMA/CCX bandwidth domains,
KleidiAI/AMX kernels) and most of it has no GPU meaning. What does carry over is the
**backend-agnostic serving layer**: admission control, the execution budget, envelope metrics
(`safe_play_start`, `max_gap`, stall rates), the soak/screen harnesses and the KPI contract.
Those measure a server, not a CPU, and should be pointed at the CUDA server unchanged — that is
the honest way to find out whether a GPU box is actually better per euro.

Explicitly NOT transferable: specs 11A/12 and the Arm decoder cohort work, since a GPU-resident
decoder replaces that component rather than tuning it.

## 4. Order of work on the rented box

1. Build `make cuda`, run `--caps`, `--self-test`, `--dispatch-map` (baseline, no PR).
2. Merge PR #29 with `gh pr merge 29 --merge` (never a local squash: that reassigns authorship
   away from the contributor) and re-run its `decoder_convT_packed` self-test.
3. Verify the TQ-7 refusal fires and that `--backend cuda` without `--prefork` is unaffected.
4. Re-run REPRO-1's A/B/B probe against the CUDA server, fused talker on, to close the
   steering-only caveat on the delta-prefill guard.
5. Only then: P3/P4/P5 measurements, and any CUDA-only feature flag that those justify.

## 5. GPU box results — 2026-09-15 (A100 80GB, then RTX PRO 6000 Blackwell)

The first box was a spot instance that was reclaimed mid-session; the campaign restarted on an
on-demand RTX PRO 6000 Blackwell (sm_120, 96 GB, CUDA 12.8, 30 cores). Findings below are from
whichever box is named; the three code fixes were reproduced on both.

### 5.1 Fixes that came out of it

* **CUDA did not compile at all on this branch.** `qwen_admission_health_t` (`qwen_tts.h`) uses
  the C11 `_Atomic`, and nvcc compiles the `.cu` units as C++, where it is not a keyword:
  8 errors in `qwen_tts_cuda_talker.cu`, `make cuda` dead on the first unit. Fixed (`079c3c2`)
  by giving C++ a plain-typed, layout-compatible definition. **The CI has no CUDA job**, which
  is why a branch with a broken GPU backend was a merge candidate.
* **`QWEN_CUDA_CONVDEC=1` aborted** with `munmap_chunk(): invalid pointer` in
  `sd_stream_st_body`. `conv_decoder_forward()` freed with plain `free()` a buffer the caller
  had allocated with `sd_tmp_alloc()`, which hands out interior pointers of the decoder arena.
  **Not CUDA-specific**: that branch is reached only when `sd_exact_stream_enabled()` is false,
  which the default CPU build never does, so it sat latent — `QWEN_SD_WINDOWED=1` reproduces
  the abort on the CPU with no GPU present (rc=134 before, rc=0 after). Fixed (`6a06523`).
* **TQ-7 guard verified on hardware**: `--backend cuda --prefork 2` returns rc=2 with the
  intended message. It had been written blind on the dev machine.

### 5.2 PR #29 — validated, and it is not sufficient on its own

Merges clean (0 conflicts), `f4b0e5e Da3dalusCode` preserved in history. Its own new self-test
passes: `decoder_convT_packed (naive)` and `(gemm)` both at `rel = 6.278e-08`. The static
diagnosis holds — `sd_pack_convt` writes `[k][ic][oc]` and overwrites the weight pointer in
place, so CUDA always received the packed tensor while `kd_convT` read `[ic][oc][k]`.
But the PR alone does **not** make the GPU decoder usable: the arena free above still aborted
the process. Both fixes are needed.

### 5.3 CPU/CUDA parity (0.6B, seed 42, temperature 0, ryan/English — verified in the logs)

| comparison | mel_corr | verdict |
| --- | --- | --- |
| CPU CLI vs CPU server | **1.00000** | byte-identical; the old CLI/server gap is closed by the REPRO-1 fix |
| CUDA server req1 vs req2 | 0.99993 | reproducible across requests |
| streaming vs non-streaming | 0.99990 / 1.00000 | **streaming parity is sound** |
| CUDA modes among themselves (seam/dec/convdec) | 0.993 – 0.999 | decoder variants are equivalent to the seam |
| CUDA CLI vs CUDA server | 0.724 | path-dependent (matvec vs matmat) |
| **CPU vs CUDA** | **0.38 – 0.64** | different generation, both models |

The fork is introduced by the bf16 seam every CUDA mode shares, not by the decoders. It is
explained by precision: the GPU seam self-test reports `rel` ~1e-3 on Blackwell
(`matvec 1.879e-03`, `matmat 1.624e-03`) against ~1e-7 on the A100 (`matvec 2.326e-07`), while
the CPU paths agree bit-for-bit. At ~1e-3, any path difference changes the sampled trajectory.

**Listening verdict (owner, 2026-09-15): the CPU and CUDA renditions BOTH sound good.** So the
divergence is a different valid take, not degraded audio. The consequence is procedural rather
than qualitative: a mel-corr golden gate cannot validate the CUDA path against a CPU reference,
and a GPU lane needs its own reference set.

### 5.4 Speed

`QWEN_CUDA_CONVDEC=1` is the fastest CUDA mode. Server, same request: 2.60 s vs 3.86 s for the
plain seam, a third faster. CLI 0.6B: 4.03 s vs 5.03 s. `QWEN_CUDA_FUSED_TALKER` is the most
divergent mode and generated 14.08 s of audio where CPU produced 11.12 s.

### 5.5 Measurement trap worth keeping

`tests/serve_soak.py` defaults `--precision` to **int8** (`:561`). A CUDA soak launched without
`--precision default` therefore runs `--int8`, and since the backend seam is bf16-only the GPU
does nothing: measured `utilization.gpu = 0%` while the run looked healthy. The startup NOTE
added in CUDA-1 is what caught it. Any GPU serving measurement must pass `--precision default`
and be confirmed by GPU utilisation, not by the presence of a CUDA banner.
