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

## 6. GPU design: what we already have, and where the time actually goes

### 6.1 Correction: CUDA graphs are NOT missing

An earlier reading of this track listed "CUDA Graph" as a gap. That is wrong.
`qwen_tts_cuda_talker.cu` already captures and replays two graphs: one for the 28-layer
talker step (`:229`, capture/instantiate at `:380-388`) and one for the 5-layer code-predictor
step (`:720`). Both are stream-captured once and replayed with `cudaGraphLaunch`.

What is true is **where** they live. The graph-backed step runs only when all of these hold
(`qwen_tts_talker.c:689`): `QWEN_CUDA_FUSED_TALKER` is set (`main.c:1695`), the context is the
single `g_gpu_fused_owner`, and no steering vector is active. The server uses the batched
paths and never reaches it, so on a serving workload the graphs are simply not in play.

### 6.2 The per-frame host round-trip, which a graph does not fix

`qwen_cuda_talker_step()` brackets each graph launch with synchronous transfers:

    cudaMemcpy(s->x, embed, H*4, H2D)        // synchronous
    cudaMemcpy(s->d_pos, &pos, 4, H2D)       // synchronous
    cudaGraphLaunch(exec, stream)
    cudaStreamSynchronize(stream)            // full sync, every frame
    cudaMemcpy(hidden_out, s->xn, H*4, D2H)  // synchronous

So every generated frame costs three synchronous copies and one full stream synchronisation.
The graph removes kernel-launch overhead *inside* the step; it does nothing about the
host/device round trip *around* it. At ~12 Hz frame rate over a multi-second utterance this is
the dominant structural cost of the fused path, and it is the first thing to attack — not the
absence of graphs.

### 6.3 What vLLM-Omni does differently (external reference, for direction only)

vLLM-Omni serves Qwen3-TTS as a two-stage pipeline (Talker -> Code2Wav) and reports RTF 0.16
single-stream and 0.29 at concurrency 10 on an H200. Techniques it names: continuous batching
on the Talker with static batching on the vocoder; async chunking that forwards codec segments
(default 25 frames) downstream so decode overlaps generation; a **dynamic initial chunk**
(2-16 frames, sized by server load) that cut time-to-first-packet from 733 ms to 64 ms; CUDA
graphs; kernel fusion in the code predictor; and re-prefill instead of a KV cache for the code
predictor, on the grounds that its sequences reach only ~16 tokens so O(T^2) attention is
cheaper than block-table bookkeeping.

Mapping that onto this engine: continuous batching, cross-stage overlap and decoder batching we
already have; CUDA graphs we have but only on the unreachable fused path; the dynamic initial
chunk we do **not** have, and it is a scheduling policy rather than a kernel, so it would help
the CPU lane too. Our code predictor uses a KV cache (`cp_kv_max = 64`), the opposite of their
choice — worth measuring rather than assuming either way.

Sources: docs.vllm.ai/projects/vllm-omni (Qwen3-Omni TTS performance optimization; Qwen3-TTS
online serving) and the vLLM blog post of 2026-07-01.

### 6.4 Constraint on any of this work

The owner's standing rule: **the CPU lane must never regress.** Every GPU-specific change goes
behind `#ifdef QWEN_HAVE_CUDA` *and* a default-off env flag declared in
`g_qwen_reported_flags[]`, exactly as the existing `QWEN_CUDA_*` levers are. `make blas` must
not even compile the new code, and `make test-all` with the golden gate remains the regression
net. The two fixes landed today follow that shape: the TQ-7 guard and the offload NOTE are
inside the GPU `#if` and absent from the CPU build.

## 7. TF32 refuted as the cause of the CPU/CUDA divergence

Hypothesis: the hardcoded `CUBLAS_TF32_TENSOR_OP_MATH` (10-bit mantissa) explained the
backend's `rel ~1e-3` and therefore the audio fork. **Refuted by measurement** on the
RTX PRO 6000, with the mode made switchable (`QWEN_CUDA_TF32`, commit `633dbfd`):

| mode | matvec rel | matmat rel | cuda matmat |
| --- | --- | --- | --- |
| TF32 on (historical default) | 1.879e-03 | 1.624e-03 | 0.035 ms |
| TF32 off (fp32 math) | **1.879e-03** | 1.656e-03 | 0.039 ms |

Identical for matvec, marginally worse for matmat. TF32 is not the source.

**What is:** the self-test compares the CPU's bf16 kernel against the GPU's f32 GEMM, and the
two boxes differ in *CPU* capability, not GPU. The A100 host built `SIMD=portable` and reports
`bf16 dot: widen->FMA (no AVX-512-BF16)` — it widens bf16 to f32, so it agreed with the GPU to
`2.326e-07`. The RTX PRO 6000 host builds `SIMD=avx512bf16` and uses the native bf16 dot, whose
8-bit mantissa (2^-8 ~ 3.9e-3) is consistent with the observed 1.9e-3. **The GPU is the more
accurate of the two**; the reference is the coarser side.

**And precision is not the lever anyway.** On the A100, where CPU and GPU matvec agreed to
2.3e-07, the audio still diverged (mel_corr 0.518). Greedy sampling over 75-100 frames will
eventually flip an argmax on a difference of any size. So CPU/CUDA audio divergence is inherent
to running different arithmetic, not a defect to be tuned away, and it confirms §5.3: a GPU
lane needs its own reference set rather than validation against a CPU golden.

The `QWEN_CUDA_TF32` lever is kept — it costs about 11% on this matmat when disabled and is
useful for isolating precision questions — but it must not be described as a fix for parity.

## 8. Where the serving time actually goes (measured)

Mini-soaks on the CUDA streaming server (0.6B, bf16 seam + `QWEN_CUDA_CONVDEC=1`, single
process, `--prefork-threads 16 --batch-size 8`), 2 minutes each, GPU sampled throughout:

| C | RTF p50 | RTF p95 | TTFB p50/p95 | TTFA p50/p95 | safe-start p50/p95 | prebuffer p50/p95 | st@250 | st@500 | GPU median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.56-0.60 | 0.61-0.68 | 37 / 59 ms | 98-119 / 165-231 ms | 118 / 280 ms | 0.008 / 0.035 s | 1% | 0% | **15%** |
| 5 | 0.67-0.71 | 0.71-0.85 | 41 / 66 ms | 117-126 / 171-177 ms | 172 / 471 ms | 0.042 / 0.153 s | 12% | 0% | **23%** |
| 6 | 0.80-0.83 | 0.90-0.93 | 46 / 88 ms | 130-146 / 232-250 ms | 375 / 719 ms | 0.249 / 0.511 s | 44% | 2% | **14%** |

Threshold: **C4 clean, C5 soft edge, C6 past the knee.** The earlier `soak_fast` screen agrees
(C4 HEALTHY, C8 KNEE). Note these are 2-minute screens: the `per-class KPI drift` FAILs are an
under-sampling artifact by construction and must not be read as a verdict on C4.

**The GPU never exceeds 23% median (38% peak) while the server is already knee-ing.** Latency is
excellent (TTFB 37 ms, TTFA 98 ms at C4); it is sustained throughput that collapses. That is the
signature of a device waiting on the host, not of a saturated device.

The cause is structural in the seam (`qwen_tts_cuda.c:105-121`): every `matmat_bf16` call does a
synchronous H2D copy, one `cublasSgemm`, and a synchronous D2H copy. With 28 layers and roughly
seven matvecs each, that is ~200 blocking PCIe round trips per generated frame. A per-operation
offload is round-trip-bound by construction and cannot be fixed incrementally. It also stores
weights as f32 on the device after a host-side bf16->f32 expansion, doubling weight memory and
forgoing bf16 tensor cores.

**Consequence for the work order:** optimising GPU kernels, or extending the CUDA graphs, would
not move the concurrency threshold while the device idles at 15%. The fix is the resident path —
which already exists and already carries the graphs, but sits behind `QWEN_CUDA_FUSED_TALKER`
with a single `g_gpu_fused_owner` that the server's batched paths never reach (see §6.1).

## 9. QWEN_CUDA_BATCH has never worked — and the repo's own self-test says so

The resident batched GPU path (`QWEN_CUDA_FUSED_TALKER` + `QWEN_CUDA_BATCH`, reached through
`qwen_tts_serve_continuous`) is the architecture this track hoped to make the CUDA default. It
is comprehensively broken, and the breakage is **pre-existing**, not introduced by this
session's changes. `--gpu-batch-bench 8` on the RTX PRO 6000, run on the binary from before and
after the mask fix:

| | before mask fix | after mask fix |
| --- | --- | --- |
| illegal memory accesses | 10296 | 11362 |
| Talker `max abs(batched - single)` | **2.93e+01 FAIL** | **2.93e+01 FAIL** |
| CP `max abs(batched - single)` | 0.00e+00 PASS | 0.00e+00 PASS |
| Talker throughput vs single | **GAIN 0.12x** | GAIN 0.12x |
| CP throughput vs single | GAIN 0.61x | GAIN 0.60x |

Readings:

* **The batched Talker computes a different answer**, off by 29.3 absolute against the
  single-stream oracle. That is not precision drift; it is a wrong result. The code predictor,
  by contrast, is exact (0.00e+00), so the defect is confined to the Talker batch path.
* **It is also eight times slower than single-stream** (0.12x). Even repaired, batching as it
  stands is a loss, so "fix the crash and make it the default" would not have been the win it
  looked like.
* `compute-sanitizer` on the standalone self-test names the faulting kernel:
  `Invalid __global__ write of size 4 bytes at k_matmat_bf16(...)`, reached through
  `talker_body_batch -> qwen_cuda_talker_batch_step`, with an address that is not a plausible
  device pointer at all ("137866940870132 bytes before the nearest allocation"), i.e. a bad
  destination rather than an index that overshoots. 216417 errors in one run.
* The self-test prints `FAIL` unprompted. This has been visible to anyone who ran
  `--gpu-batch-bench`, which evidently nobody did.

### What this session changed, and what it did not

`c749ac0` makes the CUDA branches of `qwen_batch_talker_step_ragged` and
`batch_cp_transformer_step` forward the `active` mask, which they were dropping. Those branches
were genuinely wrong — a lane that is not stepping keeps the position of whatever request last
held the slot, and every position-indexed kernel derives an address from it — and a host-side
clamp would have been unsafe because a slot can be active-but-paused (`step_active` narrows
`active` through the lead, priority, width and decoder gates) and writing its KV would corrupt
a live request. **But the mask was not the cause of the observed failure**: the self-test
reproduces it with no server, no concurrency, and every lane active (it passes NULL). The fix
is kept as a correctness repair, not as a remedy for this symptom; no observable symptom has
been shown to depend on it.

### Next step, and it needs very little GPU time

`--gpu-batch-bench N` is a standalone, seconds-long reproducer that needs no server, no model
serving and no concurrency. Debugging the batched Talker should be done against it, not against
a soak. Start from the `k_matmat_bf16` destination pointer in `talker_body_batch`, since the
sanitizer says the write target is invalid rather than merely out of range.

Until that is fixed, `QWEN_CUDA_BATCH` must stay off, and the earlier proposal to make the
resident batched path the CUDA server default is withdrawn: today it would ship wrong audio
with the GPU idle, and it fails silently — the server still answers 200 with plausibly sized
WAVs.
