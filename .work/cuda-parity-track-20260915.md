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

> **MERGED 2026-09-16** as `79ca337`, after the arena defect below was fixed independently. The
> "not sufficient on its own" verdict in this section was true at the time and is now historical:
> both fixes are in main. Re-measured before merging on an RTX PRO 6000 Blackwell from a clean
> clone — same text, seed and voice with and without the PR — the decoder goes from **rms 137 to
> rms 1254** at identical duration, which is the white noise turning into speech. See PLAN
> CUDA-2 for the full evidence.

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

## 10. The batched Talker defect, found and fixed

`--gpu-batch-bench` bisected cleanly: exact at B<=2, broken at B>=4, with correctness and
throughput failing at the same threshold.

    before   B=1 exact 0.52x | B=2 exact 0.89x | B=4 FAIL 0.06x | B=8 FAIL 0.15x
    after    B=1 exact 0.71x | B=2 exact 1.41x | B=4 exact 2.56x | B=8 exact 5.03x

Cause: the three batched matmat kernels accumulated into `float s[QB_MAX]` through loops
bounded by the runtime batch size. With a runtime bound the compiler cannot prove the index
range, so it spilled the accumulator to local memory, which on a GPU is backed by global
memory. That one detail produced all of it at once — `max|batched - single| = 2.93e+01`,
~11k illegal memory accesses per run, and throughput collapsing to 0.06x. compute-sanitizer's
"Invalid __global__ write of size 4 bytes" inside `k_matmat_bf16`, at an address far outside
every allocation, was the spilled accumulator rather than the `Y` it appeared to target — which
is why every pointer in the batch state dumped as valid. Fixed in `c55d298` by unrolling the
per-sequence loops over the compile-time `QB_MAX` with a `b<B` guard.

Note the new trade-off this creates: `s[QB_MAX]` now lives in registers, so raising `QB_MAX`
above 8 costs registers per thread and therefore occupancy. It is no longer a free constant to
raise; it needs measuring.

Server verification, B=8 with four concurrent requests (the case that used to crash): **0
illegal memory accesses** (was 10792) and **GPU utilisation 51-71%** (was 0%). The device is
finally doing work.

## 11. But lockstep batching is worse than the naive seam for this workload

A 4-minute soak at concurrency 4 on the repaired resident path, against the earlier mini-soak of
the naive per-op seam at the same concurrency:

| | naive seam | resident batched B=8 | resident batched B=4 |
| --- | --- | --- | --- |
| RTF p50 | 0.56-0.60 | 0.79-0.80 | 0.77-0.81 |
| RTF p95 | 0.61-0.68 | 0.83-0.85 | 0.83-0.87 |
| TTFB p50 | 37 ms | 33-36 ms | 33-35 ms |
| TTFA p50 | 98-119 ms | 98-134 ms | 96-110 ms |
| safe_play_start p50/p95 | 118 / 280 ms | 419 / 600 ms | 413 / 562 ms |
| **stall@250** | **1%** | **61%** | **62%** |
| stall@500 | 0% | 0% | 0% |
| GPU | 15% | ~70% | ~70% |

Latency KPI and resource stability PASS in every arm; zero illegal accesses.

**Matching the batch size to the concurrency changes nothing** (61% vs 62%), so the wasted-lane
hypothesis is refuted and a dynamic `B_eff` would not fix this. The engine already compacts
active slots (`qwen_batch_pack_active` fills `bb->act_idx` and sets `bb->B_eff`, used by
`qwen_batch_proj`); wiring that into CUDA remains reasonable work, but it is not the cure for
these stalls. Note also that compaction is not simply "pass B_eff": the KV cache is indexed per
slot, so compacted lanes need a lane->slot map for cache addressing even while activations stay
dense.

The cause is **lockstep**: every lane advances one frame together, so a short request waits on
the longest in the group. The stall distribution says so — 62% at 250 ms and **0% at 500 ms**,
i.e. gaps that are regular and bounded rather than a heavy tail, which is the shape of a
cadence hiccup, on a bank with five length classes.

**Conclusion for the serving lane.** The batched path is now correct and 5x better in raw
throughput, but at C4 it delivers worse playback than the simple seam. It should NOT become the
CUDA server default on these numbers. What would change the picture is genuine continuous
batching, where requests join and leave without forcing a shared frame cadence — which is the
architecture vLLM-Omni describes, and a substantially larger piece of work than either the
`B_eff` compaction or raising `QB_MAX`.

## 12. Correction, and where the time actually goes

**A correction first.** Section 11 compared the naive seam at 1% stall@250 against the resident
batched path at 61%, and concluded the batched path served worse. That comparison was
confounded: the seam arm was running with `QWEN_CUDA_CONVDEC=1` and the resident arm was not,
so it measured the GPU speech decoder being on or off, not seam versus resident. Run with both
arms identical (no CONVDEC, batch-size 4, concurrency 4), the order reverses — resident 59%
stall@250 against seam 95%. The conclusion in section 11 is withdrawn.

**Per-stage profile at C4** (`QWEN_SERVE_PROFILE=1`, share of work):

| stage | seam | resident | +CONVDEC | all three |
| --- | --- | --- | --- | --- |
| talker step (batched) | 20.7% | 22.1% | 29.8% | 34.9% |
| **code predictor** | 38.7% | 32.8% | **57.0%** | **49.7%** |
| speech decode + embed | 37.8% | 42.0% | 9.7% | 11.4% |
| everything else | ~3% | ~3% | ~3% | ~4% |

**The talker is about a fifth to a third of the time.** Continuous batching of the talker —
the change that was about to be written — addresses only that slice, so even making it free
caps out around a 21-35% gain. The bulk is elsewhere, and the ordering is now measured rather
than assumed.

## 13. All three CUDA paths on: the best configuration measured, and it was never tried

Talker and CP resident+batched together WITH the GPU-resident conv decoder had not been run in
combination before; every earlier arm enabled a subset. At concurrency 4, 3-minute soaks:

| | seam | CONVDEC only | **all three** |
| --- | --- | --- | --- |
| stall@100 | 100% | 8% | **1%** |
| stall@250 | 95% | 1% | **0%** |
| stall@500 | 0% | 0% | 0% |
| safe_play_start p50/p95 | 532 / 681 ms | 124 / 291 ms | **93 / 236 ms** |
| max_gap p95 | 0.668 s | 0.562 s | **0.502 s** |
| illegal accesses | 0 | 0 | 0 |

`SOAK RESULT` is PASS for CONVDEC-only and FAIL on TTFA p95 alone for the combination, on a
3-minute screen; every other KPI passes and the playback envelope is the best of any arm.

**Consequence for the work order.** The next target is the code predictor at ~50% of the time,
not the talker scheduler. Our own stage note describes it as "15 sequential passes per frame;
re-reads its weights 16x", and vLLM-Omni independently names the same component for its two
headline optimisations (re-prefill instead of a KV cache, since its sequences reach only ~16
tokens, and fusing ~60 kernels). Two systems arriving at the same component from different
directions is the strongest signal available here.

Continuous batching of the talker remains a legitimate piece of work, but it is third in line
behind the code predictor and behind simply shipping the three-path configuration, and it
should be sized against a measured 35% ceiling rather than an assumed one.

---

## 14. The code predictor, opened up — 2026-09-16 (RTX A6000, EPYC 7402)

A second box, weaker than the A100 on both sides: an A6000 (768 GB/s) behind a 10-core EPYC
7402. Build note worth keeping: the Makefile's default `NVCC_ARCH` includes `sm_120`, which
CUDA 12.6 does not know, so a 12.6 toolkit needs `make cuda NVCC_ARCH="-gencode
arch=compute_86,code=sm_86"`.

### 14.1 The compaction commit was verified, and the existing test could not have done it

`b7b916b` (QWEN_CUDA_BATCH_COMPACT) reported PASS with the flag on and off, identical to the
last digit. The reason was not that it worked: `--gpu-batch-bench` calls the batched step with
`active=NULL`, so every lane steps and the compaction path never runs. The test proved the
flag inert, not correct.

A compaction fault is silent by construction — a lane packed to the wrong dense index reads
another request's KV and returns plausible audio — so it needs a reference that cannot itself
be wrong. `batch_mask_selftest` (884eebd) runs B independent one-wide states, each stepped
exactly on the steps where its lane was active, and hands idle lanes a wrong position and a
wrong embedding on purpose. B=8, 24 steps, 128 lane-steps compared, 64 idled, compaction on:
**0.00e+00**.

### 14.2 The measurement the previous session set up, and what it said

`QWEN_CP_PROFILE=1`, C4 soak, all three CUDA paths, 2500 frames:

```
seed 0.036 ms (0.1%)   step 13.35 ms (54.7%)   head 11.00 ms (45.1%)
```

Neither of the two hypotheses on the table was right alone; both halves mattered. The
`head` — final norm, lm_head, argmax — was not on the GPU at all.

### 14.3 Correction: the CUDA graphs were missing where it counts

§6.1 of this note recorded that CUDA graphs are not missing. That is true only of the
**single-stream** bodies. The server runs the **batched** ones, and those had no graph:

| per served frame | launches |
| --- | --- |
| `talker_body_batch` | 28 layers x 19 = 532 |
| `cp_body_batch` | 15 passes x 5 layers x 19 = 1425 |

`61d989d` caches one graph per effective lane count — the grid geometry and the by-value lane
count are fixed at capture, positions and masks stay in device buffers read at replay. Worth
**-5.0% on the Talker and -8.2% on the CP**, not the ~30% the launch arithmetic suggests:
launches are asynchronous, so most of the issue cost was already overlapping with execution.
The estimate that motivated the change was too optimistic by about 4x.

### 14.4 The head: 8.6x, and bit-identical

One lm_head is [2048 x 1024] bf16 = 4 MB; a frame walks fifteen of them; and
`qwen_argmax_matvec_bf16` takes one activation vector, so the CPU walked each matrix **once
per lane** — 240 MB per frame from DRAM at four lanes. `k_cp_head_part` (50a5735) gives one
block a slice of the vocabulary and holds every lane's activation in shared memory, so a
weight row is read once and dotted against all lanes from registers: 60 MB, on the GPU's bus.

C4 soak, four minutes each arm:

| | head off | head on |
| --- | --- | --- |
| CP step | 12.38 ms/f | 12.31 ms/f |
| CP head | 11.61 ms/f | **1.35 ms/f** |
| CP total | 24.03 ms/f | **13.70 ms/f** |
| safe_play_start p50/p95 | 1731 / 4352 ms | **1373 / 2633 ms** |
| stall@250 | 84% | **57%** |
| stall@1000 | 41% | **19%** |
| requests completed | 82 | **96** |

Codes are bit-identical: a fixed-seed temperature-0 streaming request gave the same md5 with
the head on and off, `max|diff| 0` over 6.16 s. Not assumed — the summation order differs, so
a near-tie could have resolved the other way, and it was measured because of that.

### 14.5 Refuted: the seventeen host round trips are not the cost

This note's §6.2, and the plan that followed from it, held that the CP's per-pass host
alternation — upload, launch, full sync, download, fifteen times a frame — was the thing to
fuse away. `qwen_cuda_cp_batch_bench_fused` times the same fifteen body replays back to back
with **one** sync and no copies. The answer, at B=4 and B=8:

```
fused ceiling 11.62 ms/f vs 11.53 ms/f served   (B=4)
fused ceiling 13.21 ms/f vs 13.17 ms/f served   (B=8)
```

**Zero.** The GPU is genuinely busy for the whole pass; the host is never the critical path.
Fusing the loop onto the device — the obvious big project, and the one vLLM-Omni's "fuse ~60
kernels" pointed at — would have bought nothing. Ten seconds of measurement instead of a day
of work.

### 14.6 Where the step time actually went: latency, not bandwidth

CP batched cost 0.77 / 0.72 / 0.82 ms per pass at B=2 / 4 / 8. A kernel whose cost barely
moves while arithmetic and activation traffic quadruple is limited by neither, and 139 MB in
0.82 ms is 180 GB/s on a 768 GB/s card. `k_matmat_bf16` had one warp per row issuing one
2-byte load per iteration with the next instruction consuming it. Four independent loads per
lane (f43a354), stride unchanged at 32 so both weights and activations stay fully coalesced:

| B=8 | start | + graphs | + unroll |
| --- | --- | --- | --- |
| Talker | 7.56 ms/f | 7.18 | **6.47** (-14.4%) |
| CP | 14.35 ms/f | 13.17 | **11.14** (-22.4%) |
| aggregate GAIN | 4.73x | 5.10x | **5.90x** |

One trap on the way. Summing the four products as a single expression lets the compiler build
a different reduction tree, and the batched Talker then disagreed with the single-stream
reference by 1.61e-01 — reordering, not error, but indistinguishable from a bad index in a
maximum. Accumulating in the original order is bit-identical **and faster**. The selftest now
prints the first six steps' divergence, which is what settled it: a wrong index is wrong on
step 0, a drift climbs through the residual stream.

### 14.7 What is now known about the remaining time

The head is no longer a target at 1.35 ms/f. The step is, and it is a kernel-efficiency
problem inside `k_matmat_bf16` rather than a structural one — still roughly 2.5x off the
memory roof after the unroll. The int8 and q4 batched matmats have the same shape, but the
CUDA seam is bf16-only so nothing served reaches them.

### 14.8 The harness was measuring a one-thread server

Four kernel-level wins in a row moved the client-observed metrics less and less, and the last
one gave it away: the code predictor step fell 15% (8.60 -> 7.32 ms/frame) and RTF p50 at C4
moved 4% while the stall rate did not move at all. Optimising the right thing and seeing
nothing means the constraint is elsewhere.

It was `--prefork-threads 1`, which with `--prefork 1` sizes the entire server pool. Our soak
driver passed it, and `tests/serve_soak.py` defaults it to 1 at `:563`. Every GPU soak in this
campaign — today's A6000 ladder and yesterday's A100 arm alike — ran the server with **one
engine thread on a ten-core box**.

C4, everything else held fixed:

| threads | RTF p50 | stall@250 | stall@1000 | completed | safe_play_start p50/p95 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.68 | 44% | 11% | 82 | 1092 / 2311 ms |
| 4 | 0.58 | 36% | 4% | 91 | 955 / 2172 ms |
| 8 | 0.56 | 36% | 4% | 91 | 925 / 2047 ms |

The CP step is unchanged across all three (7.37 / 7.34 / 7.31 ms/frame), so this is entirely
CPU-side work and not a GPU effect. Four threads captures it; eight adds nothing.

The engine's own default is `cpus/n` and has always been correct, so no user was ever affected
— but every GPU serving number this project has recorded understates the server by roughly this
much, and the A100 arm had more headroom at C4 than we credited it with.

This is the third harness default of the same shape, after `--precision int8` silencing the GPU
entirely. The pattern to watch for is a default that disables what is being measured while the
run still looks healthy.

### 14.9 Refuted: the lane cap was not the constraint

QB_MAX was 8, and at C10 the server queued 453 requests because two of ten could never be
admitted. Since the code predictor reads 2.24 GB of weights per frame regardless of B — the
lanes share them — wider batches should amortise it, so raising the cap looked structural.

Raising it alone made everything slower: at QB_MAX=16 the Talker went 4.81 -> 7.38 ms/frame and
the CP 10.62 -> 18.06 **at B=8**, serving the same eight lanes. `s[]` is an array of registers
sized at compile time, so sizing it at QB_MAX with a runtime guard burns QB_MAX registers
whatever B is. Templating the lane count fixes that, and then:

| per stream | Talker | CP |
| --- | --- | --- |
| B=4 | 0.99 ms | 2.05 ms |
| B=8 | 0.61 | **1.34** |
| B=12 | 0.60 | 1.43 |
| B=16 | 0.58 | 1.47 |

Eight lanes is already the optimum: past it the kernel stops being weight-bound and becomes
bound by per-lane registers. The queueing at C10 is admission working correctly against a
saturated GPU, not a cap worth raising.

The win landed at the narrow end instead, which is where a compacted server actually runs:
B=4 Talker 6.02 -> 3.97 ms/frame (-34%), CP 9.99 -> 8.18 (-18%).

### 14.10 Where the remaining headroom is, ranked by evidence

The code predictor reads 2.24 GB per audio frame and the batched kernels move it at about
209 GB/s on a 768 GB/s card. That ratio, not scheduling, is what sets the concurrency a GPU can
hold.

1. **int8 weights on the resident CUDA path — untested, halves the bytes.** `k_matmat_int8` and
   `k_matmat_q4_0` already exist here and `mvB` already selects on `s->prec`; the models are
   simply loaded bf16 for the GPU. Note the confusion that hid this: `--int8` disabling the GPU
   is a property of the bf16-only *seam*, not of the *resident* path.
2. **The matmat itself, 209 vs the 500-616 GB/s cuBLAS reaches on the same shapes.** cuBLAS is
   wired and measured but needs bf16 activations, which moved the end of speech and produced an
   18% longer utterance, so it stays off. A split-K hand kernel would keep f32: the kernel gives
   one warp per output row, so at rows=1024 it launches only 128 blocks onto 84 SMs.
3. **The speech decoder**, 11-15% of stage time and not graph-captured.

Do NOT re-open: fusing the CP loop onto the device (§14.5, measured at zero) or raising the lane
cap (§14.9).

### 14.11 Six restructurings of the batched matmat, and what survived

After the head, the attention and the graphs, the batched matmat is the whole remaining cost:
the Talker step reads 706 MB of bf16 weights (353M parameters over 28 layers) in 4.75 ms, which
is 149 GB/s on a 768 GB/s card. ncu: DRAM throughput 22%, compute throughput 36%, Waves Per SM
0.25, achieved occupancy 25% of theoretical. Nothing is saturated; it is latency-bound.

Six restructurings were written and measured. One helped.

| change | result at B=8 | verdict |
| --- | --- | --- |
| four weight loads in flight | 7.56 -> 6.47 ms/f | **kept**, bit-identical |
| block size 256 -> 64 | 4.86 -> 4.72 | **kept**, bit-identical |
| split-K over the reduction dim | 4.82 -> 4.98 | removed |
| wmma tensor-core tiles | 4.73 -> 5.13 | kept default-off (see below) |
| 16-byte vector loads | 4.73 -> 5.01 | removed |
| activations staged in shared memory | 4.72 -> 6.80 | removed |
| residual add fused into the epilogue | 4.72 -> 4.79 | removed |

Each failure says something specific, and together they say the same thing.

**Split-K** quadruples the grid but leaves each block about two unrolled iterations; the
per-block fixed cost eats the gain. Slower with CUDA graphs on AND off, so not an overlap
effect.

**Tensor cores** accelerate arithmetic that was never the constraint. The kernel is correct and
portable (wmma bf16 16x16x16, sm_80 through sm_120, guarded on `__CUDA_ARCH__` with a runtime
capability check) and is kept default-off because it carries the per-stage precision gate; it is
not kept because it is fast here.

**Wide loads** raise bytes-in-flight from 256 KB to 2 MB, past the ~460 KB a 768 GB/s card needs
to saturate, and were still slower. At eight lanes each weight load needs eight activation
loads, so 32 of every 36 load instructions are activations and every warp issues its own against
L1.

**Shared-memory staging** is the textbook fix for that and was slower by 44%, because the
arithmetic does not work at this shape: eight warps per block cover eight rows, so the staged
activation tile (B x TC floats = 16 KB) is larger than the weight tile the block consumes
(8 rows x TC bf16 = 8 KB). A real GEMM amortises by giving a block 128 rows, which needs each
warp to hold several rows in registers -- register blocking, and a different kernel.

**Fusing the residual add** removes 56 launches per Talker step and was 1.5% slower, repeatably:
the read-modify-write epilogue costs more than the launch it saves.

The reading that matters for whoever picks this up: every win today came from fixing something
identifiably absent or broken -- a head that was not on the GPU, an attention kernel with eight
barriers per key position, batched bodies with no CUDA graph, an accumulator sized at QB_MAX.
Every attempt to squeeze the already-reasonable matmat further has failed. The remaining 5x
needs either a register-blocked GEMM, which is days of work, or reduced precision, which is
measured and costly (below).

### 14.12 Per-stage precision: built, measured, and not enough

`mvB` takes `allow_reduced`, and every call in `cp_body_batch` passes 0, so the code predictor
-- the stage that picks the codes through an argmax -- stays f32 while the talker may run
reduced. vLLM-Omni draws the line in exactly the same place.

It works as designed: cuBLAS on the talker alone is worth 8% (4.73 -> 4.35 ms/frame) with the CP
still bit-identical at 0.00e+00. It is still not enough to enable. Same text, seed 42,
temperature 0, code predictor protected:

```
duration            7.44 s -> 5.60 s   (-24.7%)
log-spectrum corr   0.860
```

Protecting the code predictor does not contain the divergence, because the talker's hidden state
is what the code predictor consumes. Reduced activations anywhere in that chain are a different
generation, not a rounding difference.

### 14.13 Determinism follows the engine thread count

The server is deterministic for a FIXED thread count and not across thread counts. Same request,
`--prefork-threads 1` gives md5 7ca2fc9a618e8b47, `--prefork-threads 8` gives 2e26bd462b23d77b,
each reproducible. That is the documented CPU-path behaviour -- reduction order follows the
thread count -- now confirmed on the GPU serving path.

Two consequences. The 17% RTF win in §14.8 from sizing the pool correctly is **not
output-neutral**. And every bit-exactness claim in this campaign holds only with the thread
count fixed, which every comparison here did.

### 14.14 The speech-decoder CUDA graph: root-caused, and worth nothing

The GPU-resident conv decoder is ~150 kernel launches per call and was the last stage with no
CUDA graph, at 11-15% of serving time. vLLM-Omni captures its Code2Wav stage for this reason, so
it looked like the best remaining candidate of the "absent, not broken" kind that paid every time
today.

**What the decoder is actually called with** (one 6.16 s request, traced): lengths 1, 3, 5, 9, 13,
21, then 28 for every remaining call — a short ramp for time-to-first-audio, then a fixed window.
Thirteen calls, seven at 28. A graph per length would cover nearly everything on a long request.

**Why capture failed, and it was one line.** `cudaStreamBeginCapture` succeeded and
`cudaStreamEndCapture` then reported only "operation failed due to a previous error during
capture", which names no operation. Asking the stream its capture status after each operation
turned that into a location in one run: status Active after the conv-transpose, Invalidated
immediately after the residual **`cudaMemcpy` device-to-device**. A synchronous copy cannot be
recorded and kills the capture where it stands. cuBLAS, the obvious suspect and the thing this
body has that the Talker's does not, captures without complaint.

With both residual copies made async the capture survives the whole body and EndCapture returns
clean.

**And it buys nothing.** Five fixed-seed requests: 11.93 s with the graph, 11.94 s without. Same
reason the batched-body graphs were worth 5-8% rather than the 30% the launch count suggests --
launches are asynchronous and already overlap execution. The work was reverted rather than kept:
it also produced a different waveform, so keeping it would have meant chasing a correctness bug
for a measured zero.

Recorded because the root cause is the useful part. If the decoder is ever restructured, the
synchronous residual copy is the thing that blocks capture, and the capture-status probe is how to
find that class of bug in one run instead of by reading.

### 14.15 The 2-second admission is 99.5% host work, and the GPU has nothing to do with it

`QWEN_STAGE_TRACE` put 43% of server wall time in admission: 2.9% of iterations hold 45% of all
wall time and 95% of THAT is the prefill of a newly admitted request, which stalls the whole
batch for up to 1977 ms — precisely the 2.0 s max_gap that forces a 1.9 s safe_play_start.

The prefill's matmats route through the CUDA **seam**, not the resident batched path: the hook is
process-wide, so `qwen_matmat_bf16` lands in `qwen_cuda_matmat_bf16`. That seam keeps weights as
f32, uses `cublasSgemm`, and does a synchronous H2D and D2H on every call, which made "move the
prefill onto the resident bf16 path" look like the obvious next project.

`QWEN_CUDA_SEAM_STATS=1` priced it instead of assuming:

```
[SEAM] calls=3116  total 278 ms | H2D 23 ms (43 MB)  GEMM 186 ms  D2H 52 ms (119 MB)
admit = 53.3 s     prefill = 53.2 s     of wall = 126.3 s
```

**The seam costs 278 ms. The prefill costs 53,200 ms.** About nineteen admissions in the run, so
each prefill spends roughly 15 ms on the GPU out of 2.8 seconds: **0.5%**. The synchronous
transfers total 75 ms across the entire run.

So all three seam defects are real as descriptions and irrelevant as causes, and there is nothing
to move: the prefill's matmats are already on the GPU and already free. The 99.5% is host work —
norms, RoPE, SwiGLU and the causal O(N^2) attention over the prompt, all of which stay on the CPU.

**Consequence: safe_play_start cannot be reduced by GPU optimisation.** The two real levers are
optimising the CPU prefill, which is outside the CUDA scope and invalidates CPU baselines, or
writing a genuine GPU prefill — causal attention over N positions, plus norms and SwiGLU — exposed
as a CUDA entry point and called from `qwen_tts_talker.c` inside `#ifdef QWEN_HAVE_CUDA` with the
CPU body untouched. The second respects the scope rule but is a new kernel of a different shape
from the decode step (one sequence at N positions, not B lanes at one position each), so it is a
project, not a refinement.

### 14.16 Prefill slicing is a CPU mechanism that does not transfer to CUDA

`QWEN_PREFILL_SLICE=N` runs an admission's prefill in resumable slices, one per frame iteration,
so streaming slots are not stalled. It is the obvious answer to §14.15 and it is wrong here.

C4, three minutes each, everything else fixed:

| | slice=0 | slice=8 | slice=16 |
| --- | --- | --- | --- |
| requests completed | **91** | 56 | 70 |
| safe_play_start p50/p95 | **0.95 / 1.93 s** | 4.11 / 13.20 s | 1.24 / 6.76 s |
| stall rate @1000 ms | **4%** | 46% | 17% |
| admit_ms max | 1976 ms | **569** | 679 |
| iterations >200 ms, share of wall | 44% | 69% | 58% |

It does exactly what it was designed to do — the admission peak collapses from 1976 ms to 569 —
and everything that matters gets worse: throughput falls 38% and safe_play_start quadruples.

On the CPU the prefill competes with decode for one thread pool, and slicing lets the other slots
through. On CUDA the cost is per-CALL, so slicing into ten multiplies the fixed cost by ten
instead of spreading it. A mechanism designed for one backend can be actively harmful on another,
and the flag would have shipped on by default if it had been trusted rather than measured.

---

## 15. Next CUDA phase: a native GPU prefill, separate from decode (design note, not started)

§14.15 established that admission is 43% of server wall time, that a single prefill stalls the
batch for ~2.8 s, and that **15 ms of that is GPU**. The matmats already run on the device and are
free; the rest is host work. So the next phase is not another kernel tweak — it is giving the
prefill its own GPU path.

### 15.1 Shape

```
qwen_cuda_talker_prefill(ctx, embeds, seq_len, slot)
    RMSNorm / RoPE over N positions
    causal attention N x N              <- the piece that does not exist today
    SwiGLU / MLP
    KV population directly into the slot's device cache
  -> hand back to the normal continuous batched decode
```

It is a different kernel shape from everything written on 2026-09-16. The decode step is B lanes
at ONE position each, attending to history; this is ONE sequence at N positions, attending
causally among themselves. None of the `_b` kernels transfer unchanged.

### 15.2 Why it is worth more than it looks

The KV lands on the device already. `qwen_cuda_talker_batch_upload_slot()` exists precisely to
push a newly admitted request's KV from host to device, and a native prefill removes that
transfer as a side effect rather than as a separate optimisation.

### 15.3 Scope rule, non-negotiable

Everything new goes in `qwen_tts_cuda_talker.cu`. The call site in `qwen_tts_talker.c` is an
`#ifdef QWEN_HAVE_CUDA` block that tries the device and falls through, with the CPU body left
character-for-character unchanged — verified by stripping the ifdefs and diffing, not asserted.
CPU serving baselines cost a month and any structural edit invalidates them.

### 15.4 Estimate and PoC BEFORE implementing

This session spent seven restructurings to win two; the same discipline applies, and here the
measurement is cheap.

1. **Split the 2.8 s on the host first, with zero code change.** `perf` during a soak, or the
   regions the engine already carries (`QWEN_RGN_TK_PREFILL`, `QWEN_RGN_TK_PF_WEIGHT_PREP`) via
   `make cost-map`. If the bulk is NOT the causal attention, a GPU attention kernel is the wrong
   project and the estimate stops there.
2. **Price the GPU floor for whatever dominates.** For causal attention at N=400, 28 layers,
   16 heads, head_dim 64: roughly 18 GFLOP including both QK^T and AV, which is single-digit
   milliseconds on an A6000 even at poor efficiency. A floor that is 100x under the current cost
   is what would justify the work; a floor within 5x would not.
3. **PoC the attention kernel standalone** against the CPU prefill for one prompt, checked for
   exactness the way `--gpu-batch-bench` checks the decode path, before wiring anything into the
   server.

### 15.5 What to expect it to fix, and what it will not

It attacks the admission spike, and therefore `max_gap` and `safe_play_start` — the metrics that
did not move all session however much the decode kernels improved. It does nothing for steady-state
RTF, which is already bounded by the batched matmat and its memory roof (§14.11).


### 14.17 Concurrency ladder on a strong GPU (WIP), and a sparse-ladder mistake

RTX PRO 6000 Blackwell (97 GB), 30 cores, CUDA 12.8, 0.6B, all resident paths,
`--prefork-threads 8`, length-varied corpus. Eight two-minute screens. Full table with every
metric and the exact command in `docs/cuda-performance.md`.

| | C2 | C4 | C8 | C9 | C10 | **C11** | C12 | C16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTF p50 | 0.19 | 0.25 | 0.36 | 0.41 | 0.41 | **0.47** | 0.69 | 0.86 |
| safe_play_start p50/p95 | 49/110 | 59/125 | 94/189 | 102/251 | 107/226 | **104/237** | 183/564 | 408/749 ms |
| max_gap p95 | 0.21 | 0.29 | 0.38 | 0.44 | 0.45 | **0.46** | 0.61 | 0.75 s |
| stall @100 / @250 ms | 0/0% | 0/0% | 0/0% | 2/0% | 2/0% | **2/0%** | 22/12% | 94/41% |
| requests / 2 min | 119 | 158 | 220 | 213 | 236 | **238** | 178 | 192 |
| audio-s / wall-s | 10.5 | 16.0 | 22.2 | 21.4 | **24.1** | 23.0 | 17.4 | 18.8 |

**C8 to C11 is a plateau and C12 is a cliff.** Across the plateau stalls stay at 0% from 250 ms
upward, safe_play_start stays near 105/240 ms, and throughput climbs to 238. One rung further
costs 12% of stalls at 250 ms, doubles safe_play_start p95, and drops throughput by a quarter.

**The mistake, recorded because it was mine and it was already published.** The first pass ran
C2/C4/C8/C12/C16 and concluded that C8 was the last clean rung and that throughput peaked there
— which agreed exactly with the batched kernel's per-stream optimum at B=8 (§14.9). Two
independent routes to the same number is the strongest kind of evidence, and that is why it went
into three documents and was pushed. Filling in C9, C10 and C11 dissolved it: the plateau runs to
C11 and throughput peaks at C10-C11, so the agreement was an artifact of where the ladder
happened to stop. A doubling ladder establishes "the highest rung MEASURED clean" and never "the
highest rung that is clean", and the elegance of a coincidence is not evidence for it.

**The admission stall of §14.15 does not appear here.** max_gap p95 is 0.21-0.46 s across the
whole plateau against ~2.0 s on the A6000, and the prefill is host work: this box has 30 cores
against 10. That supports the diagnosis and says where the GPU-prefill project of §15 would pay
most — boxes whose CPU is small relative to their GPU, the usual shape of a rented inference box.
