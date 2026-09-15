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
