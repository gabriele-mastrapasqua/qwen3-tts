# Current Plan

`ENGINEERING.md` is normative. This file is the short task queue; reasoning and reviewed
evidence live in the linked `.work/*.md` addenda.

## Mission

Build a CPU TTS server that starts quickly, continuously feeds a real 1x player without
repeated starvation, keeps realtime headroom, protects established streams from new
arrivals and from unrelated slow clients, and only then maximizes sustainable concurrency
and cost per stream. The qualification process discovers the highest concurrency that
satisfies the complete streaming envelope; C4 is not a required operating point.
Rationale and evidence: `.work/professional-streaming-architecture.md`.

## Current trusted state

- Host: GCP c4-standard-24 (12 physical cores, SMT off), 1.7B INT8, decoder Design D
  INT8 AMX with persistent packs, 2x6 prefork, engine-owned pool, batch cap 2.
- Current conservative short q8/threshold2 control: C2/C3 are GOOD; C4 is MARGINAL
  (`STREAM_RTF` p50/p95 0.793/0.856, required-prebuffer p95 596 ms, stall@500 25%).
  `STREAM_RTF < 1` is capacity, not a continuous playback proof.
- Fused-residual Design-D candidate: pooled five-minute C4 SOAK passed the hard stream
  gate in all four windows (`STREAM_RTF` p50/p95 0.8304/0.8933, TTFA p95 526 ms,
  safe-play-start p95 917 ms, zero errors/rejects/timeouts); preferred `<=0.90` was
  missed in one window and per-class p95 was under-sampled. The flag remains default-off.
- Post-C4 capacity screen: C5 is the first NOT STREAMABLE point under the complete
  envelope (STREAM p95 0.852 but TTFA p95 4.44 s and safe-play-start p95 4.60 s); C6
  shows the same failure. F2 causally decomposed the C5 tail: under cap 2, three full-wave
  requests waited 3.6–4.9 s before parent `accept()` and >97% of client-to-first-PCM
  elapsed before engine admission; `--max-queue 0` converted the same overload to 3
  immediate parent-side 503s. A secondary child/engine queue + prefill term remains, but
  is sub-second. Do not advertise C5/C6 as realtime capacity. Detail:
  `.work/p4-prefork-admission-bound-20260907.md`, `.work/f2-c5-startup-decomposition-20260908.md`.
- CT-1 confirms prebuffer follows quantum (q8 ~0.7 s p95 in short SOAK; q32 ~2.5 s)
  while RTF changes less. q32 is rejected as a production streaming policy.
- Decoder MACs already run on real AMX with wide N; its wall is glue (im2col, quantization,
  ~41 rendezvous and ~110 BLAS calls per call, snake, tails). AMX can touch at most
  ~10-20 % of request wall; more tile tasks regressed (M split rejected).
- Per-worker effective batch ~1.1-1.3 at C4: Talker/CP run as DRAM-bound B=1 GEMV, weights
  read per worker. The 1x12/batch-4 probe measured decode-burst coupling, not Talker
  batching, and is not evidence against a single engine.
- Inline prefill stalls every established stream 108-240 ms per admission; a slow client
  blocks its worker's engine thread (blocking writes, no send timeout).
- LS-4 utilization-aware admission was falsified on the same host: thresholds 40/60/80 ms
  admitted all tested fifth arrivals, but established STREAM_RTF p95 stayed 0.985-1.004,
  stall@250 was 50%, and post-admission max-gap p95 reached 653-704 ms. Keep the
  diagnostic default-off; cap2/q4 remains the reference. Detail:
  `.work/ls4-utilization-aware-admission-20260908.md`.
- Harness (2026-09-07): one metric core `tests/playback_sim.py` with per-request
  safe_play_start, fixed-buffer stall rates, max_gap, coalesced-read share; marks are
  client-observed. Batched synchronous streams now publish the header at admission and
  accepted sockets set `TCP_NODELAY`; PCM writes remain synchronous unless OUT is enabled.
  Detail: `.work/mt4-transport-boundary-20260907.md`.

### Current 8-core AMX product decision

- The current 8-physical-core reference is `1x8@0-7`, SMT off, Design-D INT8,
  fused residual, warm strip, q4, engine pool and fail-fast admission. The
  strict `amx-product` profile now explicitly enables the official known-text
  SL-1 layout (`QWEN_TTS_STREAM_LAYOUT=1`); ICL/clone and live incremental text
  are outside this lane.
- 1.7B: **C2/cap2 is the highest full-envelope GOOD point**. C3 is a healthy
  short/isolated-bank screen but not a full production point because the
  corrected C3 SOAK still has tail drift and pooled STREAM_RTF p95 just over
  one. Detail: `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- 0.6B: **C3/cap3 is the highest full-envelope GOOD point** on this host. C4 is
  a non-promoted screen; C5 is the first clearly bad short-bank point. Detail:
  `.work/gcp-c4-highcpu16-amx-product-capacity-20260908.md`.
- These are playback-aware product points, not a claim that the 8-core host can
  sustain C4 or that low-rate Poisson probes establish an economic rate. Cost
  per good stream remains UNKNOWN without grounded pricing.

## Immediate priorities

### Legacy CPU / v2 portability audit — detail: `.work/legacy-cpu-v2-audit-20260916.md`

- [x] LEGACY-CPU-0 Read-only architecture audit at `15a5850`: reconstructed model load,
      quantization/packing, ISA dispatch, Talker/CP/decoder, GEMV/GEMM crossover and v2
      serving. Current evidence: AVX2 has batched INT8/Q4 emulation but no native INT8/Q4
      GEMV; AVX-512 without VNNI has no dedicated integer matrix family; dotprod-only ARM
      has native GEMV but SDOT matmat is opt-in and KAI currently requires i8mm. M1
      `--caps`, `--dispatch-map` and `--self-test` were rerun on a clean rebuilt HEAD.
      No execution code was changed.
- [ ] LEGACY-CPU-1..VALID-1 Execute the linked plan in order: dispatch/profile truth,
      measured AVX2/AVX512-no-VNNI and dotprod/NEON baselines, one kernel family per A/B,
      then backend-aware v2 policy. Do not infer legacy capacity from VNNI/AMX/KleidiAI
      results.

### MAXIMUM PRIORITY — P0 sustained closed-loop soak regression — detail: `.work/arm-sustained-soak-regression-20260913.md`

This is the current serving blocker before any new headline concurrency claim. Keep the
benchmark families separate: TRUE-WAVE / parallel capacity, sustained closed-loop capacity,
and realistic arrival-load capacity (for example POISSON). A true-wave result is never a
sustained qualification.

- [x] ARM-SOAK-0 Reproduce and classify the regression (2026-09-13): C6/C7/C8 mini-soaks
      reproduce it in roughly 2–5 minutes; zero crashes, rejects, request timeouts and
      obvious functional failures, but STREAM_RTF p95 is around/above 1, safe-start rises,
      and stall@250/@500 becomes material. The baseline does not show systematic
      window-by-window growth; memory, threads, FDs and scratch remain stable. Long STAGE
      iterations are 271–364 ms and the problematic samples contain roughly 171–196 ms of
      admission/prefill while decoder time is negligible. This is currently a
      serving/scheduling/QoS problem, not a proven allocator leak.
- [x] ARM-SOAK-0a Prefill-helper A/B: control vs `QWEN_PREFILL_HELPER=1`, same Arm v2
      all-on C8/4x8 1.7B closed-loop run. The helper removed inline prefill from STAGE but
      worsened STREAM p95 `1.041 -> 1.098`, safe-start p95 `717 -> 1129 ms`, stall@250
      `20.1% -> 46.8%` and stall@500 `2.8% -> 10.1%`; errors/rejects/timeouts stayed
      `0/0/0`. **REJECTED** as a treatment; the broader admission/prefill resource-
      interference hypothesis is **PARTIALLY CONFIRMED**, because the helper still uses
      the shared engine pool and adds contention. Keep helper default-off.
- [x] ARM-SOAK-1 **P0-A playback-first admission guard** (2026-09-13): implemented as a
      reversible `QWEN_ADMISSION_GUARD` policy in the continuous batched lane and tested
      at the same all-on C8 closed-loop point. With a 400 ms ready-audio target, control →
      guard changed STREAM_RTF p95 `1.061 -> 1.053`, stall@250 `25.5% -> 13.8%`, but
      stall@500 `4.7% -> 6.5%`, safe-start p95 `776 -> 3066 ms`, TTFA p95 `241 ->
      2805 ms`, and completed requests `173 -> 147`; errors/rejects/timeouts stayed
      `0/0/0`, with stable threads/RSS/FDs. The guard recorded 111 deferred admissions
      versus 87 immediate admissions; defer p50 was roughly 1.0–1.6 s per worker, p95
      2.8–3.0 s, and the largest observed defer was 19.4 s. **PARTIALLY CONFIRMED**:
      it reduces short playback gaps, but the current policy over-protects by making
      fresh admission latency and throughput unacceptable, while the 500 ms tail remains
      unsafe. Do not promote this target or start slicing yet; tune/retest the admission
      decision as the next P0 experiment. The guard is intentionally limited to the
      continuous batched queue; `JOB_SINGLE` is not covered by this A/B.
- [x] ARM-SOAK-2 **P0-B cooperative prefill slices** (2026-09-13): the existing
      token-range path was exercised at `QWEN_PREFILL_SLICE=24` and `48` on the same C8
      setup, with `[ADMSLICE]` proof in both runs. It did not provide bounded playback
      occupancy: nonzero `prefill_ms` was invariant at about `83.5/96.6 ms` p50/p95 for
      24 and `82.4/96.7 ms` for 48; STREAM/safe-start/stall tails did not improve enough
      to offset the cost. **REJECTED for playback protection in this configuration**;
      do not test 72. The next blocker is the source of the approximately 80–100 ms
      non-preemptible floor (kernel/layer/setup granularity), to be located before any
      finer-grained preemption or genuine resource-isolation A/B.
- [x] ARM-SOAK-2a **P0-B finer prefill checkpoint**: the short component trace found no
      single 90 ms kernel. A 28-layer range is serial and non-yielding: per-layer total
      `2.92/3.29 ms` p50/p95, range total `83.24/88.43 ms`, setup/finalize approximately
      zero. The layer-level checkpoint is implemented behind `QWEN_PREFILL_LAYER_SLICE`
      and the prescribed C8 sweep ran with token slice/guard/helper off on the same new
      binary. Occupancy scaled as intended: baseline `84.9/119.5 ms`, layer 1 single-
      digit (sampled ~3 ms), layer 2 `5.9/10.6 ms`, layer 4 `11.7/20.1 ms` nonzero
      `prefill_ms` p50/p95. Results were:

      | layer group | TTFA p95 | safe-start p95 | STREAM p95 | stall@250 | stall@500 | completed |
      |---:|---:|---:|---:|---:|---:|---:|
      | 0 (baseline) | 223 ms | 846 ms | 1.049 | 30.4% | 6.3% | 79 |
      | 1 | 2460 ms | 3380 ms | 1.105 | 25.8% | 11.3% | 62 |
      | 2 | 1208 ms | 2079 ms | 1.076 | **11.8%** | **2.9%** | 68 |
      | 4 | **757 ms** | **1872 ms** | 1.131 | 22.9% | 11.4% | 70 |

      All points had zero errors/rejects/timeouts. **PARTIALLY CONFIRMED**: true layer
      checkpoints remove the ~85 ms non-preemptible floor and layer=2 materially protects
      playback, but the one-pending-admission design turns that protection into admission
      starvation; no point is promoted as the production default. Do not micro-tune 3/5/6
      layers. Next discriminator is genuine admission/playback resource isolation (or an
      equivalent bounded admission policy) rather than another token/layer sweep.
- [ ] ARM-SOAK-3 **P0-C explicit temporal/QoS budget**: cap admission wall time and return
      control to active generation when the playback budget is at risk; start from the
      layer=2 checkpoint, not the rejected token-range slice.
- [x] ARM-SOAK-4 **P1 resource isolation** (2026-09-13): tested a genuine per-worker CPU
      partition on Graviton5 C8/4x8, with layer=2 and no shared-pool helper. The treatment
      reserved CPU 7/15/23/31 for admission and left engine masks 0-6/8-14/16-22/24-30;
      helper binding was observed on each reserved CPU, OpenBLAS stayed at one thread and
      the resource sample stayed at 216 threads with no RSS/FD growth. It did not retain
      the layer=2 playback benefit: control vs isolate-1CPU STREAM p95 `1.070 -> 1.156`,
      stall@250 `31.4% -> 30.3%`, stall@500 `8.6% -> 24.4%`, safe-start p95
      `2.60 -> 3.63 s`, while TTFA p95 improved `1.39 -> 0.78 s`; completed requests
      were `51 -> 49`, with zero errors/rejects/timeouts in both. Actual 2-layer groups
      were initially 7-14 ms but under sustained load commonly 31-47 ms, with full
      14-group prefill about 425-580 ms on the one reserved CPU. **REJECTED for this
      1-CPU partition**: isolation was real, but it did not protect playback and is
      underprovisioned for admission. Do not call the architecture fixed; no 2-CPU
      follow-up is justified because continuity did not improve strongly.
- [x] ARM-SOAK-4a **Graviton5 topology/bandwidth discriminator** (2026-09-13): the 32-core
      Neoverse-V3 guest exposes one socket, one NUMA node and one 48 MiB L3 shared by
      CPUs 0-31. Existing `membw` measured full-host read 13.1/102.3/159.0/160.8 GB/s
      at 1/8/16/32 threads. The real Talker INT8 GEMV measured 1x8 at 8.82 ms/159.9
      GB/s, 2x8 simultaneously at 19.82--22.19 ms/63.5--71.1 GB/s per worker, and
      4x8 at 38.56--40.10 ms/35.1--36.5 GB/s per worker (aggregate ~143.4 GB/s).
      The same 4x8 spread layout reached ~152.7 GB/s, only ~6.5% better. The follow-up
      measured isolated 1x6/1x16 references and simultaneous 2x16 at 17.25--19.07 ms,
      73.9--81.7 GB/s per worker, **155.6 GB/s aggregate**, plus 4x6 at 35.45--37.06 ms,
      38.0--39.8 GB/s per worker, **155.4 GB/s aggregate**. Thus 2x16 is the next serving
      baseline (4x6 is a close spare-core fallback); 1x32 is only a single-worker point.
      **CONFIRMED**: cross-worker shared-cache/memory/fabric contention is material; 4x8
      is not four independent Turin-like bandwidth domains. No production soak or
      scheduler change was started; next is a short C6/C8 closed-loop check on 2x16.
- [x] ARM-SOAK-4b **Graviton5 serving-width/affinity sweep** (2026-09-13): reused the
      same five-repetition `roof_matvec` primitive, with isolated references 1x6
      `10.50 ms/134.2 GB/s` and 1x16 `10.52 ms/134.0 GB/s`. Simultaneous 2x16 reached
      `17.25--19.07 ms`, `73.9--81.7 GB/s` per worker and `155.6 GB/s` aggregate;
      4x6 reached `35.45--37.06 ms`, `38.0--39.8 GB/s` per worker and `155.4 GB/s`;
      contiguous 4x8 was `143.4 GB/s`, spread 4x8 `152.7 GB/s`. **2x16 wins** the
      concurrent shape screen; 4x6 is a close spare-core fallback. Next: short C6/C8
      closed-loop validation on 2x16, with no admission-policy tuning before that check.
- [x] ARM-SOAK-4c **Axion cross-host GEMV control** (2026-09-13): the 32-core Neoverse-V2
      Axion guest also exposes one NUMA node and one shared L3, but the same kernel gives
      1x8 `8.98 ms/156.9 GB/s`, 2x8 `10.22--10.23 ms/137.8--137.9 GB/s` per worker
      (`275.7 GB/s` aggregate), and 4x8 `15.97--16.49 ms/85.5--88.3 GB/s` per worker
      (`347.8 GB/s` aggregate). Per-worker slowdown is only ~1.81x at 4x8 versus ~4.45x
      on Graviton5, whose aggregate does not scale. **GRAVITON5-SPECIFIC CONTENTION
      STRONGLY CONFIRMED**; no Axion engine soak was run. Keep 2x16 as the next G5
      serving candidate, but treat the fabric/cache issue as a hardware-shape constraint
      that scheduler tuning alone cannot remove.
- [x] ARM-SOAK-4d **Graviton4 cross-host GEMV control** (2026-09-13): the AWS spot guest
      reports 32 Neoverse-V2 cores, one NUMA node and one 36 MiB shared L3. The same
      unchanged benchmark measured 1x8 `8.26 ms/170.7 GB/s`, 2x8 `9.65--10.07 ms` and
      `139.9--146.1 GB/s` per worker (`286.0 GB/s` aggregate), and 4x8 `12.91--12.99 ms`
      and `108.5--109.2 GB/s` per worker (`435.1 GB/s` aggregate). Per-worker slowdown is
      ~1.57x at 4x8 versus ~4.45x on Graviton5. **GRAVITON5-SPECIFIC CONTENTION STRONGLY
      CONFIRMED by two ARM controls**; no G4 engine soak or spread sweep was run.
- [x] ARM-SOAK-4e **Graviton4 short closed-loop curiosity screen** (2026-09-13): with
      the explicit Arm-v2 all-on environment and unchanged 4x8 server, 1.7B/C8 completed
      131 requests in 2 minutes with STREAM p95 `0.680`, safe-start p95 `230 ms`,
      stall@250/500 `0%/0%`, and zero errors/rejects/timeouts (**PASS**). The 0.6B/C16
      screen completed 128 with zero functional errors but STREAM p95 `1.084`, safe-start
      p95 `1.876 s`, stall@250/500 `46.8%/14.7%` (**FAIL playback**). These are diagnostic
      screens only, not G4 qualifications or capacity promotion.
- [ ] ARM-SOAK-8 **P0 decoder-lane lifecycle/QoS** (2026-09-14): on the scalable 4x8
      Arm control, C10 reaches the playback knee at B=3. Synchronous mailbox waiting was
      causal; bounded async removes that wait but leaves pause/resume tails. The first
      default-off urgency ordering A/B (`QWEN_SD_SCHED=urgency`) retained every runnable
      slot and only changed decoder enqueue order; it was **REJECTED** (STREAM p95 1.077,
      stall@250/500 14.0%/6.1%, TTFA p95 335 ms). Trace shows repeated slot pause/resume
      and mostly singleton decoder groups, not a permanently lost slot. Active baseline
      diagnostic: event-only READY→ENQUEUE→START→DONE→COMPLETE_SEEN→RESUME→NEXT_PROGRESS
      timing attributes the residual tail to lane queue/dispatch plus paired cohort work:
      B3 enqueue→start p95 `213.77 ms` vs B2 `55.56 ms`; pair compute p95 `272.05 ms`
      vs singleton `86.33 ms`; completion-seen and resume-progress stay ~35–40 ms p95.
      **CONFIRMED**: do not tune urgency or completion polling. Singleton-only (`MULTISLOT=1`)
      kept all controls fixed and moved B3 enqueue→start p95 to `135.13 ms`, B3 compute p95
      to `83.49 ms`, STREAM p95 `1.058→0.829`, stall@250/500 `10.0%/3.6%→0%/0%`,
      TTFA p95 `312→159 ms`, safe-start p95 `1049→415 ms`, completions `120→126`.
      Paired cohorts are causally poisoning B3. Static audit of the two decoder paths plus
      an isolated `decode_quantum_bench` A/B then closed the "why": `ng>1` selects a second,
      separately written decoder implementation, and on Graviton4 at the product quantum a
      cohort costs **1.63x two sequential singletons** (chunk 4: group 1 `40.6 ms`, group 2
      `81.0 ms` at `MULTISLOT=0` vs `132.1 ms` at `MULTISLOT=2`; group 1 identical across
      arms). The one clearly accidental difference found statically -- the ragged path never
      sets `g_sd_arena`, so ~40 multi-MB `posix_memalign`/`free` cycles per call replace the
      per-slot bump arena -- was **REFUTED** by a zero-code allocator arm (0% at chunk 4,
      ~5% at chunk 8). **CONFIRMED: the pair is intrinsically expensive on the Arm ragged
      path, and it is a pure loss at B=2 too, at zero queue pressure.** Option B is closed;
      no ragged-kernel project is justified by this evidence. Next is option A, dynamic
      cohort admission, implemented default-off as `QWEN_SD_COHORT_MAX_B` (cohorts only
      while `n_active <= N`, decided once per frame turn).
      **A/B RESULT (2026-09-14), KEEP default-off:** at C10 the cap removed the knee --
      STREAM p95 `1.035 -> 0.847`, safe-start p95 `1032 -> 504 ms`, stall@250/@500
      `11%/2% -> 0%/0%`, TTFA p95 flat `301.7 -> 307.8 ms`, 0/0/0 errors, resources PASS;
      the trace proves the mechanism fired and nothing else moved (B=3 cohorts `460 -> 0`,
      B=2 cohorts preserved `32 -> 108`, B=3 enqueue->start p95 `213.59 -> 134.93 ms`,
      reproducing the singleton-only arm's 135.13). Cost: completions `133 -> 122` (~-5%
      throughput). At C8 it is correctly **inert** (every delta inside single-window noise;
      per-worker occupancy there is essentially B=2, so only 8 B=3 cohorts existed to cap).
      NOT promoted: one measured window per arm, `SOAK RESULT PARTIAL`, no per-class KPI,
      no audio gate. **Open decision, do not skip to a slack formula:** microbench and the
      C8 server trace both say a cohort loses at B=2 too (B=2 pair compute p95 ~252-264 ms
      vs singleton ~86-89 ms), so the next one-change A/B is `COHORT_MAX_B=1` vs `=2`; if
      1 wins, the cohort mechanism has no operating point on the Arm ragged path and should
      be retired there rather than tuned.
      **Architectural review (2026-09-14, G4 box, no code change):** three independent
      decompositions agree the penalty is one kernel, not the ragged structure -- cost map
      puts 100% of the +49.9 ms in `conv_stack`; `QWEN_SD_PHASE` puts +42.5 ms in res1 and
      +7.1 ms in res2 (the two `qwen_conv1d_int8_v2_multi` call sites) with convt/transformer
      at parity or better; `objdump` shows `sd_dconv_multi_worker` spilling its runtime-indexed
      accumulators (40 q-stores / 33 q-loads around 30 `sdot`) where the single kernel has 0.
      The penalty is ~1.6x sequential in **every** regime (chunk 1/2/4/8: 1.59/1.65/1.63/1.71;
      S=3: 1.61/1.67); the mechanism's sharing ceiling from weight sizes and measured
      bandwidth is a few ms per pair on G4 or G5; the DL-4 unit-cost gate was never recorded
      as passed and Turin never isolated the cohort. Eight attempts to falsify retirement all
      failed. **Answer: no server condition where ng>1 wins; one concrete fix exists (S==2
      named-accumulator specialisation, exact by self-test) but its best case is parity, so it
      is not a reason to keep cohorts. Decide with `COHORT_MAX_B=1` vs `=2` at C8 then C10
      (pre-registered rule in the .work file; a null result counts against the cohort); if no
      win above noise, express retirement as `QWEN_SD_MULTISLOT=0` in the Arm profile.**
      x86/Turin untouched (`MULTISLOT=2` stays); the VNNI twin shares the array structure
      and deserves the same three-cell microbench when a Turin box is next rented.
      **Decision A/B run (2026-09-14): `COHORT_MAX_B=2` vs `=1`, C8 then C10, same binary.**
      Mechanism proven (B=2 cohorts 82/56 in cap 2, 0 in cap 1; pair compute p95 ~260-270 ms
      vs singleton ~87 ms). Pre-registered rule NOT met: STREAM p95 C8 0.625 vs **0.612**
      (cap 1 better), C10 0.831 vs 0.835 (tie); stalls 0/0 everywhere; completions +4 cap 2
      at C8, +5 cap 1 at C10; TTFA/safe-start deltas ~20-45 ms flip sign between points --
      noise at the predicted ~1.5% effect size. **VERDICT: RETIRE cohorts on Arm.**
      Recommended expression, NOT applied yet (qualification-level): `QWEN_SD_MULTISLOT=0` in
      the Arm all-on profile; `COHORT_MAX_B` stays a default-off diagnostic; Turin/VNNI
      untouched. Next: multi-window soak + per-class KPI on that profile before promotion.
      **PRODUCTION DECISION APPLIED + QUALIFIED (2026-09-15).** `QWEN_SD_MULTISLOT` set to `0`
      in `configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json` only (Turin/Axion/arm-product
      untouched; `COHORT_MAX_B` stays a default-off diagnostic in no profile). Full qualification
      on the G4 box, profile as committed: caps/self-test/dispatch-map/strict preflight PASS
      (`multislot_active:false`, `feature_status.multislot:"VALID FALLBACK"`,
      `per-item-int8-dotprod`); **C8 30-min SOAK PASS** (3805 done, 0/0/0, STREAM p95 0.562,
      safe-start p95 179 ms, stall@250/@500 0%/0%, per-class + latency + resource PASS) and
      **C10 30-min SOAK PASS** (3892 done, 0/0/0, STREAM p95 0.805, safe-start p95 361 ms,
      stall@250/@500 0%/0%, all KPI PASS). Paired against the previous private measurement
      generation on the same host/commit with cohorts ON: C8 STREAM p95 `0.801 -> 0.562`, completions +13.9%;
      **C10 `OVER LIMIT -> PASS`**, STREAM p95 `1.04 -> 0.805`, safe-start `723 -> 361 ms`,
      stall@250 `19.3% -> 0%`, completions +19.9%. **0.6B sustained recommendation moves
      C8 -> C10 (+25% density).** Only TTFA p95 at C8 regressed (+25.6 ms); at C10 it improved
      96 ms. No regression attributable to singleton decode, so cohort tuning stays closed.
      Detail: `.work/arm-sustained-soak-regression-20260913.md`.
- [x] ARM-SOAK-11 **`make soak-fast` — adaptive knee SCREEN in the suite** (2026-09-15):
      `tests/soak_fast.py` drives the canonical `serve_soak.py` at 30 s warm-up + 2x90 s per
      point, classifies CLEAR/HEALTHY/KNEE against a pre-fixed rule, stops at the knee and
      names the point worth a 30-minute run; `screen_summary.json` carries
      `"is_qualification": false`. Registered in `docs/BENCHMARKING.md` (tool table + new step
      **H2**). Screen-vs-soak calibration is OPEN: the one direct same-point pair (Axion 1.7B
      C12 cohorts-ON, 30-min `0.67` vs 180 s screen `0.653`) says screens are approximately
      faithful; the G4 "one step optimistic" reading was interpolated, not measured. Pick the
      highest CLEAR anyway — it costs nothing. **RESOLVED 2026-09-15: screens are faithful.**
      Same-point pairs: G4 0.6B C12 screen `0.829` vs 30-min `0.8364`; G4 1.7B C10 screen
      `0.830` vs 30-min `0.8307`; Axion 1.7B C12 screen `0.653` vs previous 30-min `0.67`.
      The "one step optimistic" reading compared different points and is withdrawn.
- [x] ARM-SOAK-12 **G4 knee screens** (2026-09-15, production profile `MULTISLOT=0`):
      0.6B C11 CLEAR `0.803`, **C12 CLEAR `0.829`**, C13 **KNEE `1.013`**; 1.7B C8 CLEAR
      `0.698`, **C10 CLEAR `0.830`**, C11 HEALTHY `0.829`, C12 HEALTHY `0.842`. Canonical
      30-minute qualification launched at **0.6B C12** and **1.7B C10**. Screens are not
      qualifications and are not reportable operating points.
- [x] ARM-SOAK-13 **G4 v2 canonical qualification — both points PASS** (2026-09-15):
      **0.6B C12** (3962 done, 0/0/0, STREAM p95 `0.836`, TTFA p95 198 ms, safe-start p95
      356 ms, stall@250/@500 `0%/0%`) and **1.7B C10** (3665 done, 0/0/0, STREAM p95 `0.831`,
      TTFA p95 207 ms, safe-start p95 364 ms, stall@250/@500 `0%/0%`); latency, per-class (5)
      and resource KPI PASS on both, threads flat at 92. Operating points move **C8/C8 ->
      0.6B C12 (+50%) / 1.7B C10 (+25%)**. Caveat to carry into any report: `stall@100` is
      22-23% at these densities, so the points are safe for a >= 250 ms client prebuffer
      (consistent with safe-start p95 356/364 ms), not for a 100 ms one.
- [x] ARM-SOAK-9 **CLOSING ITEM A — DONE (2026-09-15): qualified operating point published into
      the G4 profile.** `configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json`:
      `objective.preferred_concurrency` `"unspecified" -> "0.6B C12; 1.7B C10 (per checkpoint
      size)"`, `concurrency_range` `[1,16] -> [1,12]`, a new
      `objective.preferred_concurrency_evidence` carrying both 30-minute soak results and the
      >= 250 ms prebuffer assumption, and the stale `parity.notes` sentence ("does not promote a
      concurrency point before the host-specific audio and soak gates pass") replaced by the
      promoted points. Original item text:
      After the 30-minute canonical qualifications at the screened winning points, update
      `configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json`: `profile.objective.preferred_concurrency`
      (today `"unspecified"`) and the `server` block / notes must carry the qualified point per
      model size, with the soak evidence path in the `why`. Today the file still reads "does not
      promote a concurrency point before the host-specific audio and soak gates pass" — those
      gates have now passed for 0.6B C8 and C10, so that sentence has to be replaced by the real
      number rather than left stale. Do not promote a point that only has a SCREEN behind it.
- [x] ARM-SOAK-14 **Axion v2 qualification — both points PASS** (2026-09-15): **1.7B C16**
      (5207 done, 0/0/0, STREAM p95 `0.789`, TTFA p95 175 ms, safe-start p95 332 ms,
      stall@250/@500 `0%/0%`) and **0.6B C16** (5281 done, 0/0/0, STREAM p95 `0.845`, TTFA p95
      195 ms, safe-start p95 371 ms, `0%/0%`); all KPI PASS, threads flat at 216. 1.7B moves
      **C12 -> C16 (+33%)**; 0.6B keeps C16 but gains the sustained gate it never had (the
      previous generation recommended it on true-wave evidence alone). Knee at C18 for both.
      Audio gate 12/12 `mel_corr 1.00000` against a cohort-ON control. **Attribution caveat:**
      the paired screens say the cohort retirement alone is worth ~-17.5% safe-start / +4.6%
      completions at 1.7B C12; most of the C12 -> C16 move is that the ladder had never been
      walked past C12. Do not credit the flag with the whole jump.
- [x] ARM-SOAK-10 **CLOSING ITEM B — DONE (2026-09-15).** Axion measured, not assumed: the
      three-cell microbench gave `1.20x` (vs `1.63x` on Graviton4) and two paired serving
      screens agreed, so `axion-c4a-highcpu32-0p6b-all-on.json` moved to `QWEN_SD_MULTISLOT=0`
      with all three measurements recorded in its `why` and a `revert` condition. Every Arm
      profile now ships the per-item decoder; `turin-c8a-32c-vnni-product` stays at `2` until
      its own isolated microbench. Original item text:
      Audit after the G4 retirement: `arm-product.json` **0**, `axion-16c-ttfa.json` **0**,
      `aws-c8g-8xlarge-32c-arm-v2-all-on.json` **0** (retired 2026-09-14/15), and
      **`axion-c4a-highcpu32-0p6b-all-on.json` is the only Arm profile still at `QWEN_SD_MULTISLOT=2`**.
      x86 is out of scope: `turin-c8a-32c-vnni-product` stays `2` until its own isolated microbench.
      Two independent reasons to expect the same verdict on Axion: the pathology is in the shared
      Arm dotprod leaf (`sd_dconv_multi_worker` spills its runtime-indexed accumulators — 40 q-stores /
      33 q-loads around 30 `sdot` versus 0 in the single-slot kernel), and `arm-product.json`'s own
      `why` already records that its short 0.6B/1.7B 2-slot and 3-slot A/B was **2.8-3.9% slower**.
      **Do not flip it blind.** The cheap discriminator is ~2 minutes on an Axion box:
      `decode_quantum_bench` three cells (g1/c4 singleton, g2/c4 at `MULTISLOT=0` = two sequential
      singletons, g2/c4 at `MULTISLOT=2` = ragged cohort) with the profile env applied correctly
      — see the void-run warning above about the comma-joined `server-env` string. If it reproduces
      the ~1.6x penalty, retire there too (`QWEN_SD_MULTISLOT=0`, the preflight-valid fallback) and
      confirm with `make soak-fast` before any 30-minute run. Keep the flag, kernel and multi-slot
      self-test cases in the tree either way.
- [x] X86-COHORT-1 **ANSWERED 2026-09-15: the cohort loses on x86 too, but the change was NOT
      made.** Fresh Zen5 Turin (EPYC 9R45, VNNI, no AMX), OSS checkpoints, English bank.
      Microbench confirmed the prediction on a third host and a second ISA: chunk 4 gives
      `24.1 / 47.6 / 65.0 ms` = **1.37x**, chunk 8 **1.41x**, sequential pair exactly linear.
      So the penalty is the shared multi-kernel source shape, not the ISA. **But the serving
      evidence is mixed**, unlike Arm: at 1.7B C12 retiring the cohort zeroes stall@250
      (1.03% -> 0%) while STREAM p95 gets slightly worse (0.903 -> 0.923), and nothing on the
      ladder reached CLEAR (every point 0.90-0.99 against 0.83 for the qualified Arm points).
      **Owner's decision: stop.** `turin-c8a-32c-vnni-product` keeps `MULTISLOT=2` and its
      C12/C20 recommendations. Note these screens used OSS checkpoints and an English bank,
      not the customer workload, so they locate the knee for THIS workload and are **not**
      evidence against the earlier x86 recommendations. Revisit only if the exact `S == 2`
      named-accumulator kernel is written: x86 has the most per-call headroom left (1.37-1.41x).
      **No x86 regression from the Arm parity work** — the other thing this run had to
      establish. The Zen5 box was built from the same tree that carries every Arm change
      (`--self-test` 0 failures, `check-isa` PASS on the 23-file VNNI+AMX compile pass, VNNI
      resolved native in `--caps`), and with the cohort either ON or OFF the screened envelope
      sits in the same region the earlier x86 report described. Nothing in the Arm campaign
      moved x86 behaviour.
      Detail: `.work/arm-sustained-soak-regression-20260913.md`. Original item text: The Arm campaign
      retired `QWEN_SD_MULTISLOT` on four profiles after measuring a per-call loss; x86 still
      ships `2` on `turin-c8a-32c-vnni-product` and that value rests on weaker evidence than
      the Arm retirement now does. Two reasons to suspect it:
      (a) the VNNI multi worker has the **same source structure** that costs Arm its margin —
      `acc[3][4][2]` / `facc[3][4][2]` / `xv[3][2]` indexed by a runtime `S` and `mn`, where
      the single-slot kernel uses named registers; on Arm `objdump` showed 40 q-stores /
      33 q-loads around 30 `sdot` versus 0 loads in the single-slot twin.
      (b) the Turin cohort was promoted from a **combined** lane+RES1_V2+cohort smoke, never
      from an isolated arm; the DL-4 spec's own unit-cost gate (`conv_up` at 1 slot vs 2) is
      not recorded as passed anywhere in `.work`.
      Cost to answer: ~2 minutes. Recipe, exactly as used on both Arm hosts:
      `tests/decode_quantum_bench.c` patched for one cell, three runs at chunk 4 with the
      profile env applied as **separate assignments** (see the void-run warning: the profile
      emits one comma-joined line and `env $BASE` silently sets a single malformed variable) —
      g1/`MULTISLOT=2`, g2/`MULTISLOT=0` (two sequential singletons), g2/`MULTISLOT=2` (cohort).
      If the cohort is >= the sequential pair, disassemble `sd_dconv_multi_worker`'s VNNI twin
      to confirm the spill, then run `make soak-fast` on the Turin ladder before any 30-minute
      soak. Arm found +50%/+25%/+33% of density this way; x86 may well have some too.
      **Do not flip the Turin profile on the Arm result alone** — the Arm retirement itself was
      only taken after this host's own microbench plus two paired serving screens.
- [x] X86-COHORT-2 **CLOSED UNSTARTED 2026-09-15**: the ladder was walked far enough to see the
      knee (1.7B OFF C16, 0.6B ON past C20) and the answer did not justify a 30-minute
      qualification. Original item text: re-walk the Turin capacity ladder with
      `make soak-fast` (the C12 gate has been chased for a long time at a fixed concurrency;
      the Arm campaign showed the previous recommendation was simply below the knee on one
      host and that the ladder had never been walked past it). Then qualify only the winning
      point, and regenerate the x86 customer-facing numbers if they move.
- [ ] ARM-SOAK-5 **P1 admission concurrency cap**: test max one concurrent admission/prefill,
      accepting some fresh-request TTFA increase in exchange for existing-stream safety.
- [ ] ARM-SOAK-6 **P1 cohort preservation**: measure active cohort size, phase skew and
      batched-vs-per-item GEMM/GEMV behaviour; test bounded staggering only after the
      playback-first guard evidence.
- [ ] ARM-SOAK-7 **P2 allocation/churn audit**: inspect ragged temporaries and request
      setup/teardown only after scheduling experiments; current evidence does not justify
      a broad malloc/thread refactor.

### P0 BLOCKER — `make test-all` does not pass on main: first request differs from the rest

- [x] REPRO-1 **`test-serve-repro`: FIXED 2026-09-15.** Root cause was the cross-request
      **delta-prefill** in `qwen_tts_generate()` (`qwen_tts.c:1515-1546`): the context keeps
      `prev_input_embeds` and re-prefills only from the first position whose embedding differs,
      reusing the KV rows of the common prefix. Causally sound but NOT bit-identical — the tail
      is then computed in a shorter prefill with different GEMM tiling and accumulation order,
      and over ~96 autoregressive frames that forks the trajectory.
      **The test was accusing the wrong requests.** The CLI is the reference path and it matches
      `r2`/`r3` at `mel_corr = 1.00000`, and `r1` at `0.92744` — so `r1` was the anomaly while
      `test-serve-repro` used it as its baseline.
      **The rule was not "the first request"** but *any request whose text differs from the
      immediately preceding one*: with A, then B, then B, the first B is the divergent variant
      and only the exact repeat is correct (an exact repeat matches the whole prefix, trips
      `if (delta_start >= prefill_len) delta_start = 0` and recomputes in full).
      **The batched server was never affected** — `qwen_tts_generate_batch()` already clears
      `prev_prefill_len` per item (`qwen_tts.c:2091`), as do `generate_batch_multi` (`:2276`)
      and compose (`qwen_tts_compose.c:291`). Production serving is batched + prefork, so the
      qualified Arm/Axion/Turin operating points and every customer-facing number stand. The
      defect was confined to the single-process `--serve` path (the default when `--batch-size`
      is absent).
      **Refuted along the way:** the prefix cache (flag provably applied, output byte-identical)
      and the lazily built quantized weight packs (`QWEN_SD_INT8=0` and `QWEN_SD_RES1_V2=0` both
      leave all three WAVs byte-identical, so they do not participate). `server_prewarm` is only
      the first instance, not the cause: with `QWEN_NO_PREWARM=1` the A-then-B divergence
      survives, so disabling the pre-warm would have been a cosmetic patch over a live bug.
      **Fix:** `reset_request_state()` (`qwen_tts_server.c:908`) now clears
      `ctx->prev_prefill_len`, so the single-process server agrees with the batched server and
      with the CLI. Server-scoped; CLI and engine untouched. Gate goes from
      `ndiff=118333 (81.094%)` to `ndiff=0 (0.000%)`.
      Open follow-up on a GPU box: the fused-GPU guard at `qwen_tts.c:1529-1543` still forces
      `delta_start = 0` only when steering is active, so a fused CUDA talker without steering
      may still fork across requests. Detail: `.work/cuda-parity-track-20260915.md` §1.1.

### P0 Metric truth — detail: `.work/professional-streaming-architecture.md` E1, E8, E11, E12

- [x] MT-1 Receive-mark semantics audited; TTFB stamped independently of TTFA
      (`header_to_audio_ms`); coalesced-read share reported per run — detail:
      `.work/professional-streaming-architecture.md` E12.
- [x] MT-2 Per-request `safe_play_start`, stall_rate/stall_ms @100/250/500/1000 ms,
      max_gap; summaries in the wave and soak analyzers; `tests/test_playback_sim.py`.
- [x] MT-3 Superseded readings corrected in `docs/serving-operations.md` section 5,
      `docs/BENCHMARKING.md` sections 7-8, `ENGINEERING.md` section 9, AWS reference notes.
- [x] MT-4 Runtime transport boundary: batched streams send the header before synthesis and
      accepted sockets use `TCP_NODELAY`; server/client event ordering is proven on the
      continuous path. Per-chunk flush tracing remains optional and client marks remain
      client-observed. Detail: `.work/mt4-transport-boundary-20260907.md`.

### P0 C12 preferred gate on the frozen Turin architecture (ladder paused after bounded falsifiers) — detail: `.work/c12-win-track-20260909.md`, checkpoint: `.work/c12-win-checkpoint-20260909.md`

Goal: sustain C12 with the full streaming contract AND STREAM_RTF p95 <= 0.90 including
the short/conversational soak tails (today waves 0.82-0.85, soak pooled 0.912, short
0.959, conversational 0.914; cadence already good). One mechanism at a time against the
frozen `turin-c8a-32c-vnni-product` control; lever of record: decoder residency down ->
CP-overlap share down -> sustained tail down. Codex owns implementation; no push unless asked.

      Review 2026-09-09 (`.work/c12-architecture-review-20260909.md`, read-only): the
      remaining unit is half f32 BLAS (transformer/convnext/init ~12 ms + convt 8.6 ms,
      ~330 MB f32 weights per unit, excluded from int8 by construction) plus ~8 ms of
      copy/calloc glue; closed-loop admission (inline prefill) is the likely short-class
      tail; width is bounded out; phase-aware placement cannot help at sustained B3.
- [x] C12-WIN-1 Zero-code discriminators (2026-09-09): helper is NO-GO as an implementation
      (TTFA p95 172 -> 683 ms, safe-start 417 -> 922 ms, stall@250 appears) but it CONFIRMS
      the mechanism: with prefill off the loop the short class drops 0.966 -> 0.915 and
      pooled 0.923 -> 0.915 — inline admission is the short tail, worth ~0.05; fixed B3
      shows 54.9% decoder-overlap wall and CP 22.2 -> 36.5 ms median; spin 4096/16384 does
      not beat the 65536 control. Decoder unit 49.2 ms by cost map (conv_stack 43.3,
      transformer 5.2). Detail: `.work/c12-win-step1-3-20260909.md`.
- [x] C12-WIN-1a Pre-upsample BF16 diagnostic (2026-09-09): persistent BF16 weights and
      matmat path implemented/default-off. The bounded Turin server screen moved modestly
      (STREAM p95 .842 -> .825), but the same-generation paired audio gate failed
      (`mel_corr=.97890 < .98`) and the non-clean B3 diagnostic did not show lower decoder
      residency or CP overlap cost. Keep BF16 default-off; detail:
      `.work/c12-win-bf16-preup-20260909.md`.
- [x] C12-WIN-2 Decoder-residency falsifiers, first round (2026-09-09): BF16 pre-up,
      ConvT one-GEMM, allocation-only glue, and VNNI RES1_V2 split-input were each
      isolated; no candidate earned a serving A/B. These are IMPLEMENTATION verdicts:
      the tested BF16 arm covered the transformer only (5 ms of the unit) with bf16
      activations; the ConvT arm used a zero-expanded input panel (k× the FLOPs), not the
      proposed one-GEMM-per-layer on the un-expanded input; the glue arms were two pieces
      run separately. Untested: weight-only bf16/int8 for the conv_stack f32 weights
      (convnext pw, initial conv, convt ≈ 190 MB/unit) and the combined glue removal.
      Forensic audit: `.work/c12-win-forensic-audit-20260909.md`.
- [x] C12-WIN-2a ConvT one-GEMM falsifier (2026-09-09): exact decoder batch parity passed,
      but the expanded f32 panel made the treatment 14–32% slower across B1–B4/chunk 1–8.
      Rejected and reverted; no server A/B. Detail: `.work/c12-win-convt-one-gemm-20260909.md`.
- [x] C12-WIN-2b VNNI glue/preparation falsifiers (2026-09-09): allocation-only and
      split-input V2 were exact/parity-safe where tested but slower or neutral; both
      were reverted. The remaining alternative geometries are not justified by the
      current evidence. Detail: `.work/c12-win-glue-vnni-20260909.md`.
- [x] C12-WIN-10 Admission slicing (prefill as resumable token-range slices inside the frame
      loop). Spec: `.work/c12-win-admission-slicing-implementation.md`; local evidence
      `.work/c12-win-admission-slicing-20260910.md`. CLOSED 2026-09-10 on Turin: state parity
      CORRECT, serving behaviour a severe REGRESSION, flag stays default-off and unpromoted.
      Sliced-state parity was proven exact for every partition without a one-token slice, and
      two engine defects were found and fixed while proving it. The server A/B says the
      mechanism must not be promoted. Closed-loop C12, 10 minutes per arm, frozen profile,
      one variable: completed 1294 -> 721 (-44 %), TTFA p95 183 -> 1574 ms, STREAM_RTF p95
      0.922 -> 4.609, and 7 server request timeouts against 0 on the control. It made the
      established-stream interference it was built to remove about five times worse.
      Note what it did NOT test: all four workers report mean_slices=1.00, so with a warm
      prefix cache the admission prefill is ~1 new token (seq_len=10, prefix=9) and nothing
      was ever actually split. The damage therefore comes from the sliced-admission PATH, not
      from slicing -- most likely the one-admission-per-frame-iteration break stealing
      iterations from running streams. A first A/B attempt with the true-wave arrival model
      was void and is not cited: with a positive flag and n_active==0 the code takes the whole
      prefill in one slice by design, so a wave that releases every request into an idle
      engine cannot reach the mechanism at all. Any retry needs a redesign of the admission
      path first, plus a cold-prefix workload so real multi-slice prefills occur.
- [x] C12-WIN-11 A Conv-stack traffic: ConvT as ONE un-expanded GEMM per layer with a fused
      two-tap carry/bias epilogue. Spec: `.work/c12-win-conv-stack-implementation.md`.
      CLOSED 2026-09-10 on Turin: implementation CORRECT, effect NULL, flag stays default-off.
      Correctness passes on x86 -- self-test 0 failures with the convt_stack cases at 3e-8..1e-7
      against a 1e-5 contract, dispatch verified ON/OFF, CLI audio mel-corr 0.99962 at identical
      duration. The microbench (1.7B, 4 threads, taskset 4-7, B1-B4, 9 warm reps) shows no
      effect at the product quantum: chunk-4 deltas B1 -0.20, B2 -0.70, B3 +0.70, B4 +1.30 ms,
      and 16 of 32 cells faster -- a coin flip. A control-vs-control run of the SAME arm on the
      same binary in the same minutes measured a noise floor of -0.40..+2.70 ms at chunk 4 and
      up to 32 ms at chunk 16, so every one of those deltas is inside the noise. Note for any
      future rung: the >= 2 ms gate this spec asked for is BELOW this harness's own resolution
      at B3/B4 (noise alone is +2.7 ms there); a rung that needs to resolve 2 ms needs paired
      replicates, not a single run of each arm.
- [x] C12-WIN-12 VNNI glue as one combined change: out-of-place snake1, V2 kernel with
      (tail, tail_cols) context and residual epilogue, plain allocs, ownership transfer.
      Spec: `.work/c12-win-vnni-glue-implementation.md`. CLOSED 2026-09-10 on Turin:
      correctness REPAIRED, performance NO-GO, flag stays default-off and unpromoted.
      The first x86 `--self-test` failed 10 cases, all the `ctx+residual` contract: with
      -ffast-math the compiler re-associates the epilogue's four-term sum only when a
      residual is supplied. Not benign here, because the next residual unit re-quantises
      per position, so one ulp shifts amax and the whole position's scale: 115 LSB on a
      9550 peak end to end. Fixed in `d49aa10` by disabling reassociation for
      `sd_dconv_worker` alone -- self-test 10 failures to none, `QWEN_SD_GLUE=0/1`
      byte-identical, and the attribute costs nothing on the default path (control vs
      control -2.4..+2.9 ms, no systematic sign). With the epilogue exact the MECHANISM
      is slower than its control in 21 of 32 cells: +0.8 ms at B3 chunk 4 against a gate
      asking for -3 ms, rising to +60 ms at B4 chunk 16. The -11 ms seen before the fix
      was measured while the fused path was still free to re-associate, so it was not
      computing the same result as the control and was never a valid comparison.
- [ ] C12-WIN-3 Short-class fixed cost: only after WIN-10: ramp 1,2,4 (control) vs 1,4
      (vs 2,4 only inside the TTFA gate); short + conversational playback metrics. No q8.
- [x] C12-WIN-4 Old preparation flags: DIRECT_DWCONV/INPUT, STRIP, FUSED_RESIDUAL are
      inert on VNNI (AMX-D gated); the VNNI glue falsifiers are now closed. DIRECT_CONVT
      is superseded by the rejected one-GEMM falsifier.
- [x] C12-WIN-5 Phase-aware decoder overlap: NO-GO at the measured pinned B3 overlap
      share of 54.9%; no asymmetric Talker/CP width mechanism is justified.
- [ ] C12-WIN-6 Opportunistic B2 lane batching (optional, last): residency of 2 units vs 2
      requests, decoder off the critical path, mailbox bounded, reject on any cadence loss.
- [ ] C12-WIN-6b AWS campaign order, model matrix and gates for specs 10/11A/12:
      `.work/aws-qualification-checklist-20260910.md`. PRIMARY qualification path is
      1.7B Base OSS + Galatea qvoice (clone conditioning), SECONDARY control is 1.7B
      CustomVoice + Ryan (preset-speaker). The public ~25 MB CC0 grafts load on Base with
      `--load-voice ... --icl-only`; what is NOT yet exercised is the clone conditioning
      path through the BATCHED SERVER, which Phase A0/A confirms before any timing claim.
- [ ] C12-WIN-7 Short A/B gate per candidate (control vs one mechanism, repeated short C12
      waves, playback-aware metrics, gain > noise) before any soak.
- [ ] C12-WIN-8 Qualify the winner: C12 class waves, long+short, Poisson, overload
      unchanged, 30-min soak by class and 5-min window; STREAM p95 <= 0.90 overall and per
      class, stall@500 0, cadence targets kept; report an exact boundary rather than move the gate.
- [ ] C12-WIN-9 Capacity curve C10-C16 after the win, classified as preferred /
      mandatory-qualified / hard-capacity (never one "max C" number).
- Stop: if no target, no falsifier and no screen moves C12 above noise, hand the evidence to
      the post-Turin architecture review instead of stacking micro-optimizations.

### ARM-LINUX-V2 — parity implementation complete; optional policy qualification deferred

- [x] Arm Linux serving is at the v2 generation on `feature/arm-parity-vnni` (`6117437`).
      The implementation, exact self-tests, ISA/link checks, dispatch checks and final
      config policy are complete. The track document with the verified/unverified split
      and do-not-carry-over list remains `.work/arm-linux-v2-parity-track-20260910.md`.
      Headline finding, CONFIRMED against this tree: the decoder lane
      (`QWEN_SD_LANE_SPLIT` / `QWEN_SD_LANE_ELASTIC`) has NO ISA guard — only `__linux__` —
      so the mechanism of record on the Turin product profile ports to Arm unchanged, and no
      Arm profile sets it. The reason it was never tried is a wrong sentence in our own
      handoff, corrected 2026-09-10. Also confirmed: the five newest decoder flags have zero
      entries in `docs/feature-flags.md`, and `g_mm_gate[]` has no KleidiAI int8/bf16 rows.
      The old unpaired n=12 probe on a heterogeneous box at concurrency 2 against a 2-slot
      server remains non-evidence. The new exact-commit Axion screen is recorded below as
      a one-wave performance screen only, not as an Arm product qualification.
      Ordering: the build break above is NOT part of this track and must not wait for it.
      Progress on `feature/arm-parity-vnni` (2026-09-10): items 0 (link fix, = TQ-6), 1
      (KleidiAI gate rows), 2 (prefork plans on the inherited mask), 3 (docs + expectation
      rows), 4 (lane honours the requested engine width) and the Arm DL-4 leaf of item 8
      are implemented; the leaf passes the 20-case `--self-test` on aarch64 dotprod.
      Item 7's region body is wired on Arm through the prepared-state API that was written
      for it and never connected: Talker and CP batched regions now pack the KleidiAI LHS
      once per projection group and run the same kai_i8_task in-region.  Verified on the
      16-core Neoverse-V2: Talker region 12/12 WAV byte-identical on/off, CP region 12/12
      byte-identical, arm-product preflight VALID, dispatch gate PASS; C10 2x8 lane4
      elastic + RES1_V2 + GLUE + CONVT_STACK measures STREAM p95 0.843 against 0.939 for
      the untreated tree (WAVE screen, no SOAK yet).  RES1_V2 audio gate: 21/21 paired
      files, mel-corr min 0.9945.
      The original next list is now closed at implementation level: pre-transformer BF16
      wiring, rectangular/wide DL-4, and multi-slot DL-4 are all implemented and tested.
      The lane-team constraint is handled by the prepared-state prep/run pair (tid/nt),
      while the region/prepared-state API remains keyed on the ORIGINAL f32 weight pointer.
      DONE since: DL-4 rectangular/wide shapes (API `in_ch`/`out_ch`, any shape when the flag is
      on; two rectangular self-test cases exact / 5.6e-3); ConvNeXt pointwise pair on KAI
      int8 (`QWEN_SD_CNEXT_I8`, default off) --
      6 paired server texts mel-corr min 0.99736 / mean 0.99805, C10 0.843 -> 0.821.
      Item 1 implementation is now wired through full, streaming and ragged pre-transformer
      forwards: Arm KAI registers all persistent BF16 rows and unregisters them on teardown;
      the Neoverse-V2 smoke is functional on both 0.6B and 1.7B. The corrected Graviton5
      prepared-LHS micro A/B and paired C1 WAV gate are now PASS, while the broader BF16
      product promotion screen remains open (the implementation is default-off). Item 3 is implemented for VNNI and
      Arm SDOT with compact and production strided APIs, exact S=2/S=3 oracles, and a lane
      cohort. The Arm 2/3-slot WAVE reached group=2/3 with zero mailbox overruns, but measured
      2.8--3.9% slower on the short 0.6B/1.7B A/B, so it is also default-off. Evidence and
      remaining qualification gaps: `.work/arm-linux-v2-parity-implementation-20260911.md`.
      Exact-commit Axion FAST screen (Neoverse-V2, 2x8, short synchronized wave, custom
      1b7 model, INT8) reached C8 with lane split=4: C6/C8 STREAM p95 `.646/.716`,
      TOTAL p95 `.699/.806`, TTFA p95 `236/303 ms`, zero errors/rejects; C12/C14 are
      screen-only and miss playback headroom. Inline control was `.917/.860` STREAM p95
      at C6/C8; split=2 was slower, so no lane split is promoted in the Arm profile.
      This is not an apples-to-apples Turin claim: Turin has 32 cores and the reference
      screen uses the open 1.7B model. Turin's 4x8 screen was `.87/.87` STREAM p95 at
      C6/C8, making the Arm C6/C8 steady-state screen comparable despite half the cores;
      first-audio and full qualification still need a repeated product run.
      Arm cost map (REPORTED-MEASURED, not reproducible here): res1 is ~48 % of the upsample
      convs and the conv stack ~92 % of the decoder unit, so the missing V2 leaf aims at the
      largest single item. DO NOT chase the AMX strip/range port: it was measured first and
      discards only 0.4 % of columns at a 10-frame quantum (~5 % of residual-conv time). The
      three September AMX gaps are CLOSED on this branch; do not reopen them from the older
      cross-backend audit page. AMX lacking V2 is a dispatch-order CHOICE (Design-D precedes
      V2), not a gap.
      Follow-up 2026-09-11: `c6e6e26` shares the KleidiAI activation preparation across
      Talker/CP region workers (same prepared-state kernel); `5b03269` makes the BF16 KAI
      pre-up prepare synchronously before dispatch, removing a barrier that was unsafe for
      serial/GCD/narrowed pools. Mac build/self-test and a clean Graviton5 build/self-test
      pass; four Graviton5 C4 WAVs are byte-identical to the pre-change baseline. The
      Graviton5 4x8 all-on FAST screen is exploratory only (C8/C12 TTFA p95 257/328 ms,
      STREAM p95 0.794/0.853, zero errors/rejects); it does not close the Turin regression
      gate or qualify the Arm product profile.
- [x] ARM-LINUX-V2 item 8: the residual unit (res1/res2). VERIFIED backend map in
      `.work/arm-linux-v2-parity-track-20260910.md` section 2b. Four facts the dispatch map
      does not show: `QWEN_SD_RES1_V2` selects on SHAPE (`kernel>=1 && in_ch==out_ch &&
      !(in_ch&3)`), so it takes res2 and every square conv, not just res1 — implementing
      from the flag name builds half of it; residual fusion needs AMX, so VNNI also pays a
      separate pass (`QWEN_SD_GLUE` is the VNNI answer, default off, unqualified); AVX2 and
      AVX-512F-without-VNNI have NO int8 decoder conv at all, so the gap is three CPU
      families; and an undeclared `in_ch <= 768` gate drops every backend to f32 above it,
      AMX and VNNI included. Work: one dotprod/i8mm DL-4 leaf against the already ISA-neutral
      packing path, written to the `qwen_conv1d_int8_v2_ctx` contract. DONE on
      `feature/arm-parity-vnni`: the leaf exists for Arm dot-product and the API is now
      rectangular (`in_ch`/`out_ch`, Cp from `in_ch`), so DL-4 also takes the initial/pre
      convs and the wide channels that the v1 panel and Design-D paths cannot; --self-test
      covers both rectangular shapes and the 20 square ones. AVX2/AVX-512F-without-VNNI stay
      on the f32 fallback, so the three-family claim of this item is not delivered. The flag
      and the `decoder.res1_v2` row are re-documented but not renamed. Arm widened-path
      quality/perf promotion remains intentionally open; parity implementation and exact
      Arm/x86 build/self-test gates are complete (the VNNI kernel shares the API change).

- [x] ARM-LINUX-V2 final config TODO, completed last after the BF16/multi-slot A/B and x86
      VNNI compile/parity checks: update `configs/perf/arm-product.json` and
      `configs/perf/axion-16c-ttfa.json` with RES1_V2, lane, multi-slot and BF16 policy.
      RES1_V2 is available; BF16 pre-up and multi-slot remain explicit default-off controls
      until their separate quality/16-core qualification gates pass.

- [x] VNNI DL-4 multi-slot promotion smoke (2026-09-11): after the exact multi-slot oracle
      fix, the Turin product A/B ran three synchronized short waves at C4/C8/C12/C16 and
      two mixed short/long waves at C8/C12 against the explicit `QWEN_SD_MULTISLOT=0`
      control. All 120 requests per arm completed with zero errors/rejects; sustained
      stream/total p95 and req/s improved coherently at C8-C16. The Turin product profile
      now defaults `QWEN_SD_MULTISLOT=2`; the control profile pins 0, Arm/KleidiAI stays
      default-off pending its own 16-core/quality gate, and paired audio quality remains
      required before calling the feature qualified across products.

- [x] PRE-GRAVITON-5 Turin regression applicability gate: the Turin VNNI product/control
      A/B and the C4/C8/C12/C16 multi-slot smoke are already recorded in the Turin handoff
      and the preceding VNNI promotion work. The later commits `c6e6e26` and `5b03269` touch
      only the KleidiAI/Arm paths: on x86 VNNI the new region helpers are not selected and
      the BF16 KAI consumer is a fallback no-op. Therefore Turin does not need another run
      solely for this Arm-only delta. Reopen this gate if shared x86 kernels, threading,
      profiles, or dispatch code change. The Turin checkout remains a dirty bench checkout:
      sync only tracked source/config/commits, never models/private/WAVs, and keep the
      privacy/log/tree check in force.

- [x] GRAVITON-5 v2 mini-sweep and flow audit (2026-09-11): clean 32-core Neoverse-V3
      build/self-test/doctor passed; 4x8 was the useful topology. The exploratory all-on
      Arm v2 screen held C12/C16 at STREAM p95 `.834/.947` and TTFA p95 `347/470 ms`,
      while C18 crossed the edge (`1.514` STREAM, `529 ms` TTFA). Lane+multi-slot was the
      main gain; BF16-only was marginal and remains default-off. The marker follow-up now
      attributes the decoder panels: C2/C4 cost-map parity is still PASS, with 8/8 workers
      entered and 100% panel occupancy at both levels; the previous UNACCOUNTED row was
      instrumentation, not an inactive kernel. `conv_stack` is 93.0%/91.4% of the serve
      decoder map and pool wait is a real 16.2%/24.1% completion-wait share. The prepared
      BF16 LHS reuse is wired through full/streaming/ragged decoder paths; corrected C12
      FAST A/Bs improve STREAM/TOTAL p95 directionally in both orders with zero errors, and
      paired C1 WAVs are byte-identical. Pool-spin 0/4096/16384/65536 was noisy, so the
      Arm 65536 default remains. Scratch stats showed zero spills and grow-once/reused
      arenas. Full qualification remains open; details:
      `.work/graviton5-arm-v2-mini-sweep-20260911.md`.

- [x] GRAVFULL-1 Graviton5 qualification campaign execution (2026-09-11): the clean `arm-product`
      RES1_V2/KAI INT8 baseline was built and dispatched on the selected 4x8 topology;
      doctor, strict preflight, caps/dispatch/self-test, CPU check, paired structural
      audio, capacity waves and C4 SOAK passed with zero errors/rejects and zero fixed-
      buffer stalls. The all-on arm also passed serving/resource SOAK and the 1.7B/0.6B
      FAST ladders, but its paired mel gate failed (`0.88559` minimum vs `0.98`), so
      BF16 pre-up and multi-slot remain default-off. This closes the campaign execution,
      not the per-concurrency capacity qualification; exact evidence:
      `.work/graviton5-arm-v2-full-qualification-20260911.md`.

- [ ] GRAVFULL-2 Graviton5 per-concurrency SOAK qualification (campaign execution complete;
      strict promotion gate remains open): when the box is leased
      again, split the matrix by model. For 1.7B, run identical closed-loop SOAKs for
      control OFF and exploratory all-on at C6/C8/C12/C16; run C18 as a diagnostic edge
      only if admission remains meaningful (C4 is already covered for 1.7B). For 0.6B,
      extend the existing C1/C4/C8/C12 FAST screen through C16/C20/C22/C24+ until the
      knee, then SOAK the selected levels for both arms. Keep the current fail-fast
      admission/batch cap as the control, but add a second small-model pass with an
      explicitly raised per-worker batch/admission cap when testing C20/C22/C24; record
      the exact cap and queue policy in the manifest. A WAVE is not a capacity
      qualification: do not call any 1.7B C12/C16/C18 or 0.6B C16/C20+ level qualified
      without same-model SOAK evidence and errors/rejects=0. A c4a 32-core Arm result
      can select candidate C levels and cap settings for this pass, but cannot replace
      same-host Graviton5 evidence. All-on remains non-promotable unless paired audio
      also passes. After the current c4a campaign, if the 1.7B C12-C16 interval is
      incomplete or C16 fails at the current cap, add a fine-grained C13/C14/C15
      sweep for control and all-on. Run it first at the existing cap for comparability;
      if admission rejects are the limiting factor, repeat the selected levels with an
      explicitly raised cap and label that as a separate admission experiment.
      The c4a candidate admission experiment is now complete (2026-09-12): with
      batch-cap 8 per worker on 4x8, WAVE admission reached C32 with zero rejects for
      both 0.6B arms and C36 rejected 12; no raised-cap C20+ SOAK passed the strict
      playback/KPI gate. This selects C20 as an exploratory all-on candidate and C32
      as an admission-only candidate, not as qualification. Detail:
      `.work/c4a-arm-v2-raised-cap-report-20260912.md`.

      Graviton5 32-core all-on OSS campaign executed 2026-09-13 on clean `dc8bc48`
      (4x8, C9g.8xlarge/Neoverse-V3): complete 1.7B C1-C20 and 0.6B C1-C26
      capacity waves, short/long parallel waves, Poisson, 30-minute C12/C20 gates,
      and the additional 1.7B C10 30-minute gate. All completed requests had zero
      errors/rejects/timeouts in the measured gates. The customer-facing playback
      rule still classifies 1.7B C10 as EDGE (TOTAL_RTF p95 1.03, stall@250 9.2%),
      1.7B C12 as EDGE (1.07, 18.6%), and 0.6B C20 as EDGE (1.12, 65.8%); clean
      wave candidates are C12 and C16 respectively, not sustained qualifications.
      No extra small-model C24/C26 soak or WAV probe was needed after the full
      capacity/parallel coverage. Private evidence and the customer report remain
      outside the OSS tree; all-on is not promoted by this run.

- [x] GRAVBOX-2 AWS Graviton5 profile decision (2026-09-13): the 32-core G5
      `roof_matvec_int8` discriminator rejected 4x8 as a serving baseline (aggregate
      throughput did not scale and each worker slowed by about 4.45x). Do not retain a
      4x8 all-on profile for G5; preserve the topology evidence and test 2x16/1x32 only
      as a separate host-specific experiment if the box is rented again.

- [x] GRAVBOX-3 AWS Graviton4 all-on profile artifact (2026-09-13): created and validated
      `configs/perf/aws-c8g-8xlarge-32c-arm-v2-all-on.json` for the 32-core Neoverse-V2
      control, using the measured 4x8-friendly KAI/RES1_V2/BF16-pre-up/lane/multislot
      feature set. It is host-scoped and remains `unqualified`; the G4 C8/C12 screens
      select the candidate shape but do not replace full same-host quality and soak gates.

- [x] ARM-TOPO-1 doctor topology preflight (2026-09-13): `tools/doctor.py` now runs a
      short fixed-mask `roof_matvec_int8` 1x8 / simultaneous 2x8 / simultaneous 4x8
      discriminator on Arm Linux boxes with at least 32 online CPUs. It archives parsed
      rows and raw worker output in `arm_gemv_scaling.json` / `arm_gemv_*.txt`, prints
      the 4x8 scale and per-worker slowdown near the top of the report, and recommends
      `2x16` then `1x32` for G5-like contention. The verdict is topology-specific, not
      a claim that the whole instance is unusable. Offline doctor tests pass; an actual
      Arm run remains part of the next box preflight.

- [x] ARM-SOAK-CLEANUP engine baseline restored (2026-09-13): the rejected admission
      guard/helper, layer/token slicing, isolation and temporary tracing changes were
      removed from the engine. The source baseline is `dc8bc48`; the evidence and
      topology doctor changes remain local and unqualified until deliberately committed.

- [ ] GRAVBOX-1 GCP c4a highcpu-32 Arm candidate: record the Iowa region and quoted
      `$1.21/hour` cost, then—only after the per-model SOAKs and feature gates—derive a
      separate 32-core Arm profile from `arm-product` with the newly qualified flags.
      The box setup alone must not promote BF16 pre-up, lane, multi-slot or other
      optional features; keep the profile explicitly tied to its 32-core topology.
      A host/model-scoped all-on deployment candidate is now recorded at
      `configs/perf/axion-c4a-highcpu32-0p6b-all-on.json`: C16 preferred, C20 soft edge,
      C32 admission-only. The final clean-tree confirmation run completed 2026-09-12:
      C16 had zero errors/rejects/timeouts with STREAM p95 0.881 but missed only the
      per-class drift gate; C20 had STREAM p50/p95 1.02/1.09 and playback degradation;
      paired 0.6B audio was 0.94558–0.96306 vs 0.98. Keep the candidate policy scoped
      and `unqualified` until the numerical/audio delta is fixed or explicitly accepted.
      Detail: `.work/c4a-arm-v2-0p6b-profile-qualification-report-20260912.md`.

- [x] TURIN-POST-ARM **DONE 2026-09-15** — satisfied by the X86-COHORT-1 run on a fresh Zen5
      Turin box built from the tree that carries every Arm change. Gates: `--self-test` 0
      failures, `check-isa` PASS, `--caps` resolves VNNI/BF16 native, strict preflight valid.
      Screens compared the committed VNNI product profile against a one-variable cohort-OFF
      arm on both checkpoint sizes. **No regression attributable to the Arm parity work.**
      Deviations from the original wording, stated so the closure is auditable: the screen ran
      at **C12-C16 / C20** rather than C6/C8 (the ladder had to reach the knee to be useful),
      and used **OSS checkpoints with an English bank** rather than the customer workload, so
      it is a regression-safety screen and not a capacity claim. Detail and numbers:
      `.work/arm-sustained-soak-regression-20260913.md`.

- [ ] ARM optional-feature promotion: qualify BF16 pre-up and CNEXT-I8 as separate paired
      A/Bs. **Multi-slot is CLOSED (2026-09-15): retired on every Arm profile with a measured
      per-call loss, paired serving screens and a 1.00000 mel-corr audio gate** — this item's
      note that "the current multi-slot short A/B was negative" was right and has now been
      settled with host-specific evidence, see ARM-SOAK-10/14. BF16 pre-up and CNEXT-I8 remain
      unqualified: do not bundle them into a baseline claim without paired audio plus serving
      evidence. **Open discrepancy to resolve before any release claim that quotes them:** the
      host-scoped all-on Arm profiles set `QWEN_SD_BF16_PREUP=1`, while `docs/feature-flags.md`
      still describes that flag as "failed its x86 audio gate and stays off". Both statements
      can be true (x86 gate failed, Arm host-scoped policy enables it) but the doc does not say
      so, and a reader cannot tell. Fix the doc row or the profile, and say which.

### Deferred DECODER-XISA — converge decoder dataflow after C12-WIN

- [ ] After the Turin C12-WIN track reaches a stable checkpoint, commonize the winning
      streaming-decoder dataflow across x86 VNNI, x86 AMX and Arm/KleidiAI; keep this
      deferred and do not mix it into the current paid Turin ladder. Start with a short
      design/dataflow audit, then parity-gated leaves in this order: RES1_V2 direct causal
      convolution, common glue/materialization removal, pre-upsample BF16/INT8 matmat,
      and one-GEMM ConvT. Detail and gates: `.work/decoder-xisa-deferred-track-20260909.md`.

### Deferred QUANT-PTQ — calibration-aware quantization revisit (MEDIUM/LOW)

- [ ] Revisit lower-precision prefill and weight storage using calibration/optimization-aware
      PTQ (AutoRound-style or equivalent) instead of the engine's earlier straightforward
      conversion. Production keeps prefill in BF16 deliberately; the earlier simple INT8
      prefill and simple INT4/Q4 attempts were rejected because pronunciation and speaker character
      drifted audibly while the audio stayed otherwise valid. Those verdicts reject THOSE
      IMPLEMENTATIONS, not lower precision as a direction -- do not record "INT8 prefill" or
      "INT4" as architecturally disproven. Tracks: calibrated INT8/W8A8 prefill; quality-
      optimized Q4/INT4 or mixed precision for suitable Talker/CP/prefill regions; offline
      calibration only, with the C runtime consuming packed weights and scales and no
      training machinery; and a re-test of whether the V2 kernels change the premise. GATE:
      a performance gain is irrelevant unless pronunciation and speaker character survive against the current INT8 + BF16-prefill baseline, judged by paired audio, ASR
      and listening -- waveform/mel/duration equality is necessary and not sufficient, since
      the earlier rejections passed exactly those. Start only after the C12-WIN items and
      the report qualification work. Detail: `.work/quantization-ptq-revisit.md`.

- [x] QP-0 **AutoRound / calibration-aware rounding evaluated — CLOSED NO at 8 bits**
      (2026-09-15). The revisit note's §2 hypothesis is answered. At 8 bits the rounding
      rule has no headroom: Intel's own INT8/W8A8 table puts AutoRound **0.86 pt BELOW plain
      RTN** on Llama-3.1-8B-Instruct (70.06 vs 70.92, BF16 70.42) for ~10x the time and ~16x
      the VRAM; 8-bit weight tuning buys +0.0008/+0.0003 average; `auto_round` auto-disables
      its own scale search at `bits>=8` and recommends `iters=0`; neither AutoRound paper
      evaluates 8 bits across five versions; Intel publishes 58 int4 models and **one** int8,
      built with tuning off. Independently corroborated by Dettmers arXiv 2212.09720 App. C.3
      ("No scaling improvements for 6 to 8-bit models"), ZeroQuant-V2 (<0.05 ppl), the Qwen3
      quantization study, and llama.cpp discarding the imatrix at `q8_0`. **Also excluded:**
      our granularity is already the INT8 hardware maximum (per-output-channel weights x
      per-token dynamic activations = AutoRound's own `INT8` preset), and the BF16->INT8
      prefill speed ceiling is **2.0x on every ISA we run** (Arm N2/V1/V2, AMX, Zen4/5,
      M4 SME) against the **1.8x we already measure** -- there is no second speedup behind a
      better quantizer. AutoRound stays live and valuable at **2-4 bits only** (track 2).
      Redirection, ideas backlog and citations:
      `.work/quant-prefill-int8-analysis-20260915.md`.
- [ ] QP-1 **Activation-range profile (do this first, blocks QP-3/4/5).** Per linear layer,
      per token position, `max/median` ratio, across languages, with and without a voice
      prefix, prefix vs generated positions. ~20 lines of C behind a flag, no default change.
      Hypothesis under test: the SwiGLU activation-spike signature (arXiv 2405.14428) on the
      `down_proj` input, concentrated on BOS/newline/apostrophe -- tokens that live in the
      text prefix the PREFILL carries and that the acoustic-token DECODE never sees. If the
      signature is absent, QP-3/4/5 lose their rationale and the track needs a new hypothesis.
- [ ] QP-2 **Distance gate before any candidate.** Teacher-forced `KL(bf16 || int8)` per
      decode step plus flip rate, on the BF16 token stream, on LONG utterances, PER LANGUAGE,
      at temperature > 0; reuse `tools/quant/fakequant_cp.py` + `tests/quant_ladder.py`
      (references: int8 79.4 %, int4 46.3 %). Rationale: arXiv 2407.09141 -- aggregate
      accuracy and perplexity are structurally blind to the damage that matters, distance
      metrics are not. This does **not** replace the ear/ASR gate in the parent note; it makes
      it affordable by filtering candidates before a listener spends time on them.
      ⚠️ The `mel-corr 0.39-0.60` figure in `docs/runtime-map-c8a-c4.md` is uninformative on
      its own (trajectory divergence of a sampled AR model, not damage). The INT8-prefill
      rejection was an EAR verdict and it stands -- do not re-open the path on the metric.
- [ ] QP-3 **Per-K-block activation quantization in the prefill** (B=32, then 128) instead of
      one absmax over the whole K per token. No calibration, no new format; reuses the per-32
      machinery already written for Q4_0. Confines an outlier channel to its own block instead
      of crushing the token's whole row; overhead O(1/B). Cheapest real candidate.
- [ ] QP-4 **QFeP: first N prefix tokens in BF16, INT8 from there.** Removes the spike tokens
      by construction, and covers the attention-sink token that Mix-Quant's
      attention-concentration defence (arXiv 2605.20315) does not reach.
- [ ] QP-5 **QFeM: exclude the 1-3 worst layers** from the INT8 prefill, selected by QP-1's
      max/median ratio, `down_proj` first. Static and AMX/VNNI-friendly, unlike LLM.int8()
      dynamic column decomposition (which breaks tiling and is rejected).
- [ ] QP-6 **SmoothQuant alpha-sweep folded into RMSNorm / `v_proj` / `up_proj`** -- verify
      foldability against our block graph first. Free at runtime if it folds, but the first
      idea needing calibration data: start ONLY if QP-3/4/5 fall short.
- [ ] QP-7 **KV-seam control arm**: INT8 prefill everywhere except the K/V projections.
      Demoted from hypothesis to control (Mix-Quant rejects KV poisoning as the mechanism);
      cheap enough to run inside the QP-3 A/B.
- [ ] QP-8 **`iq4_nl` revisit — belongs to the INT4 track, not this one.** Same 4.5 bpw and
      the same 18-byte block as our Q4_0, a 16-entry LUT; QErr 1.10 % vs Q4_0's 1.84 %; our
      own measurement recorded +8.8 pt on the CP with the kernel parked. AutoRound **cannot**
      emit it; llama.cpp can. Already named as the cheap follow-up in `docs/quant-sub4.md` §5.
- [ ] QP-9 **Offline quantizer via `ggml_quantize_chunk()`** if the INT4 track restarts: link
      `ggml-quants.c`, feed an imatrix (diagonal of the activation second moment), consume
      `q4_K`/`iq4_xs`/`q6_K` blocks in our own kernels. No Python, no GGUF parsing.
      Side finding worth reading regardless: auto-round ships a Qwen3-TTS GGUF converter
      (`export_to_gguf/conversion/qwen3tts.py`) documenting an independent llama.cpp mapping
      of our model's structure.

### P1 Cadence truth (current binary, Tier A only) — detail: `.work/p1-cadence-truth-20260907.md`

- [x] CT-1 Quantum discriminator at C3/C4, including gang-off control; q32 is rejected.
- [x] CT-2 Decoder intercept/slope and `[SDPHASE]` attribution; SQ-1 remains GO.
- [x] CT-3 Inline admission interference measured with matched control; LS-4 remains P3.
- [x] CT-4 Talker B1/B2 measured; EO-2 remains viable (B2/B1 step ratio ~1.10).
- [x] CT-5 C2/C3/C4 playback envelope: GOOD / GOOD / MARGINAL.

### P2 Small-quantum decoder — CLOSED checkpoint: `.work/p2-checkpoint-20260907.md`

- [x] SQ-1 Bounded warm range slice: newly produced columns use direct INT8 A preparation
      and persistent Design-D B packs in the serving reference. The complete
      strip → snake → conv1 → snake → conv2 → residual executor is not implemented and
      moves to AR-1 as an architectural candidate.
- [x] SQ-2 Bounded fixed-cost audit: default-off slices cover direct streaming/ragged
      ConvT, depthwise and warm-input preparation; fused residual remains a candidate.
      Direct one-row gather/quantization and BLAS-C residual were rejected and reverted.
      Details: `.work/p2-checkpoint-20260907.md` and the linked experiment addenda.
- [x] SQ-3 Decoder AMX reachability and scoped accounting recorded; the four whole-request
      quantities are not fabricated where the current evidence has no valid denominator.
      Detail: `.work/p2-checkpoint-20260907.md`.


### AR-2 reviewed order — CLOSED docs checkpoint

- [x] AR-1 implementation audit and AR-1b external/model supplement are frozen against
      the P2 HEAD: `.work/ar1-post-p2-architecture-review-20260907.md`,
      `.work/ar1-codex-implementation-audit-20260907.md`,
      `.work/ar1b-external-research-supplement-20260907.md`.
- [x] AR-2 verified the official known-text dual-track layout against the C prompt/step
      path and froze the implementation order: `.work/ar2-sl1-semantics-20260907.md`.
      Prefix-cache reuse is not resumable prefill; q1/q2/q4 are smaller complete decoder
      calls, not intra-call preemption; whole-request AMX wall remains UNKNOWN.

### P3 Serving cadence and first-play

- [x] OUT-1/OUT-2 Bounded per-stream PCM queue, detached non-blocking writer, byte/memory
      cap, timeout, cancellation/disconnect semantics and slow/stopped-reader tests are
      implemented behind `QWEN_SERVER_ASYNC_OUTPUT=1`; C1/C2 path, matched C3/C4 Tier-A
      integration, byte-identical audio and slow-reader gates pass. It remains default-off:
      the C3/C4 wave shows no material KPI change, and longer-concurrency thread/memory
      qualification is still open. Engine enqueue and transport-write timestamps remain
      distinct. Detail: `.work/stream-output-isolation-20260907.md`.
- [x] SL-1 Known-text official dual-track layout implemented behind
      QWEN_TTS_STREAM_LAYOUT=1 and carried through CLI, batch and continuous-server
      admission paths. The known-text Ryan/English lane passed current-generation
      structural/audio, prefill-scaling and 8-core server interference gates and is
      explicit in `amx-product`; ICL/clone and live incremental text remain outside
      the lane. Detail: `.work/sl1-known-text-stream-layout-20260907.md` and
      `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- [x] LS-1 Minimal credit-gate skeleton implemented and falsified at C3/C4 behind
      `QWEN_STREAM_LEAD_GATE=1`: first audio remains eligible, but hard suppression at a
      250 ms target parks ~95.8% of checks, lowers useful worker work and does not improve
      stall rates. Keep default-off; do not add EDF/LS-2 on this realization without a new
      mechanism. Detail: `.work/playback-lead-gate-fc-20260907.md`.
- [x] LS-3' Small complete decoder calls at safe existing boundaries; q1/q2/q4/q8 floor
      established in a Tier-A C3/C4 screen. q1 is rejected; q2/q4/q8 remain policy
      candidates and no intra-call preemption is claimed. Detail:
      `.work/decoder-quantum-floor-20260907.md`.
- [ ] LS-2 Lead-feedback steady-state quantum: first chunk remains one frame, bounded lead
      window, explicit minimum efficient quantum; q8 remains the upper control until proven.
- [ ] PF-1 Residual fixed-prompt chunked prefill only if a retained ICL/reference or
      non-streaming mode still leaves a genuinely long prefix after SL-1. It is not a
      blocker for the current known-text 1.7B product point; the cloned-context
      helper/LOW falsifier is rejected as a serving substitute. Detail:
      `.work/prefill-helper-c34-20260907.md`; do not confuse it with live text.
- [x] LS-4 Bounded utilization-aware third-slot admission falsifier: the parent health
      predicate was implemented behind `QWEN_ADMIT_UTIL`, but all predeclared 40/60/80 ms
      thresholds damaged the established-four playback envelope despite making the fifth
      request interactive. Keep default-off; do not run a local threshold qualification.
      Cap2/q4 fail-fast remains the control. Detail:
      `.work/ls4-utilization-aware-admission-20260908.md`.

### P4 Overlap and decoder structural cost

- [x] Same-pool decoder consumer tested and rejected: `QWEN_DECODER_THREAD=1` on the
      engine pool caused C4 STREAM_RTF p95 `0.847 -> 1.296`, TTFA p95 `174 -> 1126 ms`
      and max-gap p95 `511 -> 1286 ms`; it observed `group=1` and did not preserve the
      inline decoder batching path. Keep default-off; detail:
      `.work/p4-same-pool-decoder-20260907.md`.
- [x] Fused residual Design-D epilogue passed the CLI byte/audio gate, a short server
      A/B in both per-slot and ragged forms, and a pooled five-minute mixed-bank C4 SOAK:
      short A/B STREAM_RTF p95 `0.831 -> 0.788`; SOAK p95 `0.8933` with zero errors and
      hard p95 `<1` in every window. Promote as an isolated **default-off** candidate;
      per-class p95 remains under-sampled. C5/C6 screens fail startup/safe-start despite
      STREAM p95 <1. Detail:
      `.work/p4-fused-residual-20260907.md`.
- [x] F1 fused-residual × quantum screen (2026-09-08): fused-on q4 is the next C4
      playback/realtime reference candidate (STREAM_RTF p95 `0.868`, prebuffer p95
      `201 ms`, stall@250 `0%`); q8 remains the higher-throughput control and q2 misses
      the preferred STREAM p95 target. Three-wave screen only; not a qualification, and
      no causal fused-vs-off frontier shift was isolated. Detail:
      `.work/f1-fused-quantum-20260908.md`.
- [x] F-cap3 C5 capacity screen (2026-09-08): cap 3 accepted the fifth-request
      wave without the multi-second parent-backlog tail, but cap-3 C5 failed the
      realtime promotion gate (`STREAM_RTF` p95 `0.969`, fifth-launch proxy `1.028`,
      stall@250 `13.3%`). Cap 2/q4 remains the reference; established-four causal
      impact is UNKNOWN because the short run used true simultaneous waves. No C6.
      Detail: `.work/f-cap3-c5-capacity-20260908.md`.
- [ ] DL-1 4+4 intra-CCX decoder lane, default-off `QWEN_SD_LANE_SPLIT=N` (2026-09-09):
      the worker mask is split into a STEP part (engine pool: Talker/CP/prefill) and a
      DECODER part (private pinned team, never the engine pool or its submit lock); the
      frame loop enqueues one bounded decoder unit per slot and blocks only when that
      slot needs another quantum while its unit is in flight (lead <= 1 quantum). Built
      from the single-CCX lane law (`T(B) = 40 + 13.5·B` ms, decoder 9.7 ms per slot,
      Talker+CP saturate the CCX at 2-4 threads) and the L3 contention falsifier (+12 %).
      A/B: one worker on one CCX, 1.7B, fixed text, q4, SL-1, inline vs lane at B2/B3/B4
      (+B5 if B4 is healthy). **GO**: B3 STREAM p95 <= 0.85, B4 <= 0.92, stall@250 = 0,
      no lifecycle/correctness issue; **strong GO**: B4 <= 0.90 without TTFA/prebuffer
      regression; **FAIL**: < 10 % better than inline at B3/B4, or Talker/CP inflation
      erases the overlap, or the mailbox recreates equivalent blocking, or lifecycle is
      unsafe. PASS -> 4x8 host screen at C8/C12/C16; FAIL -> stop, use the measured
      split to decide whether res1/VNNI decoder work is the next lever. Same task:
      `vnni-bf16-product` lane (native bf16 prefill; the f32 pin of `vnni-product` is a
      backend-selection defect) and `QWEN_POOL_SPIN=65536` promoted in the VNNI product
      lanes (measured 2x16 C8 0.893 -> 0.808). **A/B done 2026-09-09: NOT GO, not FAIL** —
      iteration wall matched the prediction (B3 64 ms, B4 72 ms; decoder-call spikes gone,
      stall@250 at B4 100 % -> 0 %) but STREAM p95 B3 0.871 / B4 0.997 miss the gate: the
      4-thread STEP side inflated Talker+CP by +27-29 % (per-slot region sections, ~8 ms per
      slot) and the 2 s clip pays the pipeline's fixed latency (+0.05 STREAM, +30-64 ms
      TTFA). Kept default-off; no host screen. Next lever per the split: the step side
      (5+3 / 6+2 split, long-bank A/B), not res1. **5+3 and 6+2 run 2026-09-09: both
      worse than 4+4 (fixed B4 1.113 / 1.364; long B4 1.015 / 1.263) — the decoder needs
      >= 4 cores to stay hidden at B4 and the step side gains only 3-7 ms from 5-6
      threads; no host screen; 4+4 is the allocation of record, architecture promoted,
      allocation not. DL-2 elastic 8<->4+4 (`QWEN_SD_LANE_ELASTIC=1`, pool width capped only
      while a decoder unit is in flight, preallocated per-slot handoff) run the same day:
      fixed B4 0.987 vs static 0.997, long B4 0.895 vs 0.906, Talker+CP 69.5 vs 69.8 ms —
      the static-partition tax is NOT the cause; the step is slowed ~2x only while the
      decoder unit runs (CP loses L3 residency to the decoder's f32 activations). Next
      lever: the decoder unit's cache footprint, measured by CP ms during overlap.**
      DL-3 falsifiers (2026-09-10): sub-quantum decode, direct ConvT/dwconv/input, NTA
      weight prefetch, hot lane workers, q8 — none moves the CP-in-overlap tax (35-38 ms
      vs 23.5); q8 reaches long B4 0.864 but at prebuffer 806 ms / stall@250 100 %. The tax
      is ~+20 ms per overlapped iteration whatever the decoder does; only the overlap
      share (decoder time on 4 cores, 15-16 ms/frame) scales it. **Next: DL-4 = res1/conv
      kernel efficiency on the lane (fewer weight re-reads, no separate f32 panel),
      metric = decoder unit ms on 4 threads and overlap share.** DL-4 built
      (`QWEN_SD_RES1_V2=1`, direct dilated conv, per-position quant, 4x4 register tile,
      weights read once per time block): res1 1.72x, unit 64 -> 50 ms, overlap share
      48 -> 39 %, lane B4 long 0.869 / fixed 0.918 (gate met); **4x8 host screen: C12
      STREAM p95 0.80-0.81 prebuffer 247 ms stall@250 0 %, C16 0.88 long / 0.92 short
      prebuffer 360 ms stall@250 0 % — twice the inline C8. Screen only: next = SOAK
      C12/C16 with a qualified profile and the V2 numerics ear/mel-qualified.** Detail:
      `.work/dl1-decoder-lane-split-20260909.md`. **Qualification sprint 2026-09-09
      (revision 28d6436, frozen `turin-c8a-32c-vnni-product`, control `-control`):** V2
      quality automated PASS (52 paired files, mel-corr >= 0.9948, ASR CER equal, wav_qc
      equal; ear verdict pending on the Mac listening set); **C12 QUALIFIED for the
      mandatory contract** in every class (waves STREAM p95 0.82-0.85, prebuffer p95
      ~260 ms, safe-start 467 ms, stall@250/@500 0; 30-min SOAK 2205 req 0 errors, pooled
      STREAM p95 0.912, TTFA p95 170, resources/drift PASS) with the preferred 0.90 gate
      missed only by the short (0.959) and conversational (0.914) classes under closed-loop
      soak; Poisson 1.5/2.5 req/s TTFA p95 172/175 ms; overload fail-fast works (per-worker
      cap). **C16 NOT RUN**: the spot host was reclaimed before Phase D. Handoff:
      `.work/turin-vnni-final-handoff-20260909.md`.
- [x] TQ-1 C16 density qualification (2026-09-09, on-demand c8a.8xlarge, revision e1b1ec7):
      waves STREAM p95 0.91-0.96, 30-min soak FAIL (pooled p95 1.004, short 1.045, 596
      per-worker rejects, 111 broken-pipe errors) — C16 = hard-capacity boundary, not a
      product point. Sweep C10-C16 + 10-min soaks C10/C11: knee at C13 (first B4 worker);
      **preferred C11** (pooled soak 0.886; short class alone 0.917, and 0.905 at C10),
      **mandatory-qualified C12**, **hard capacity C16**. Handoff §3.
- [ ] TQ-2 Fail-fast boundary: at a full host rejects surface as TCP resets / broken pipes
      instead of a 503 (4 of 28 in the C12 Poisson run, 111 of 707 in the C16 soak) — the reject path must drain the
      request before closing; also record that rejection is per worker (cap 4): C20 sent 8
      rejects with 16 host slots. Gate: 0 resets over >= 100 rejects, reject count = C-16
      for a simultaneous wave when the parent balances.
- [ ] TQ-8 Leading silence before speech: a measured ~0.5 s of dead air ahead of the first
      voiced frame on a 1.7B-class checkpoint (median 0.50 s over 36 files) against 0.06 s on
      a 0.6B-class one (52 files), consistent across every text class. It is not covered by
      any latency metric we gate on: what a caller experiences is TTFA PLUS the lead-in, so
      ~740 ms against ~186+60 ms. That is larger than anything the C12-WIN decoder ladder was
      chasing, and the ladder delivered nothing. CAUSE NOT ESTABLISHED -- model-emitted silent
      frames or an engine/prompt artefact are both open, and checkpoint size is confounded
      with training data. FIRST STEP is the discriminator, not a fix: run the same
      energy-envelope pass on the OPEN 1.7B and 0.6B models, same bank and settings; ~10
      minutes, CLI is enough. Only if it is model-side does a bounded, default-off leading
      trim make sense, gated on `safe_play_start` rather than TTFA and checked against the
      streaming decoder's continuity contract. Detail:
      `.work/leading-silence-perceived-latency-20260910.md`.
### CUDA parity track — detail: `.work/cuda-parity-track-20260915.md`

Opened 2026-09-15, before renting a GPU box, so instance time goes to verification rather than
discovery. Owner's order: **fixes first, then the parity analysis, then any CUDA-only flags.**

- [x] CUDA-1 Misleading offload banner. The seam in `qwen_tts_backend.h` carries `matvec_bf16`
      and `matmat_bf16` only, so `--backend cuda --int8` (or `--int4`) offloads NOTHING while
      the startup line claimed it did. `main.c` now prints an explicit NOTE naming the resident
      paths instead. Inside the GPU `#if`; the CPU build does not compile it.
- [ ] CUDA-2 PR #29 (`Da3dalusCode`, "Fix noise from the CUDA speech decoder with packed
      ConvTranspose weights"). **Diagnosis confirmed statically**: `sd_pack_convt`
      (`qwen_tts_speech_decoder.c:229`) writes `[k][ic][oc]`, the CPU oracle
      `causal_conv_transpose1d_naive:295` reads that same layout, and the packing overwrites the
      weight pointer **in place** (`:1321`, `:1338`) — so CUDA always received the packed tensor
      while `kd_convT` read it as `[ic][oc][k]`. Unconditional on the CUDA decoder path, not an
      edge case. The PR also adds a `decoder_convT_packed` self-test. MERGE IT WITH
      `gh pr merge 29 --merge` — never a local `git merge --squash` + commit, which reassigns
      authorship away from the contributor.
- [x] CUDA-8 **Batched GPU Talker: wrong results, illegal memory accesses and 0.12x
      throughput — FIXED.** `--gpu-batch-bench` bisected it cleanly: exact at B<=2, broken at
      B>=4, with correctness and speed failing at the same threshold. The three batched matmat
      kernels accumulated into `float s[QB_MAX]` through loops bounded by the runtime batch
      size; with a runtime bound the compiler spilled the accumulator to local memory, which on
      a GPU is backed by global memory. That one detail produced `max|batched-single| = 2.93e+01`
      (the engine's own gate printed FAIL), ~11k illegal accesses per run, and a collapse to
      0.06x. compute-sanitizer's "Invalid __global__ write" inside `k_matmat_bf16` at an address
      far outside every allocation was the spilled accumulator, not the `Y` it appeared to
      target — which is why every pointer in the batch state dumped as valid.
      Fixed in `c55d298` by unrolling the per-sequence loops over the compile-time `QB_MAX`.
      After: exact at every B, **5.03x at B=8** (33x better), zero illegal accesses, and the
      server case that used to crash now runs with the GPU at 51-71% instead of 0%.
      NOTE the new trade-off: `s[QB_MAX]` now lives in registers, so raising `QB_MAX` above 8
      costs registers and occupancy. It is no longer a free constant.
- [x] CUDA-9 **Best measured serving configuration: all three CUDA paths on together.**
      `QWEN_CUDA_FUSED_TALKER=1 QWEN_CUDA_BATCH=1 QWEN_CUDA_CONVDEC=1` with `--backend cuda`,
      single process, had never been run in combination — every earlier arm enabled a subset.
      At C4, 3-minute soaks: stall@100 1%, **stall@250 0%**, safe_play_start 93/236 ms,
      max_gap p95 0.502 s, zero illegal accesses — against 100% / 95% / 532/681 ms for the
      plain seam. Owner confirmed by ear that the audio captured **under load** at C4 is good.
      Not yet a qualification: these are 3-minute screens, and `--precision default` is
      mandatory (see CUDA-11).
- [x] CUDA-10 **The code predictor, opened up (A6000, 2026-09-16).** `QWEN_CP_PROFILE` split
      it almost evenly: GPU transformer passes 54.7%, head 45.1%, seed 0.1%. Three changes,
      all bit-identical (`0.00e+00` on batched-vs-single, CP, and partial occupancy):
      **(a)** CUDA graphs for the **batched** bodies — the ones the server runs had none, only
      the single-stream ones did, so a frame issued ~1950 launches (-5.0% talker, -8.2% CP);
      **(b)** the head — final norm, lm_head, argmax — moved to the GPU, where a weight row is
      read once for all lanes instead of once per lane: **11.61 -> 1.35 ms/frame**, stall@250
      84% -> 57%, 82 -> 96 requests in the same four minutes;
      **(c)** four weight loads in flight in `k_matmat_bf16`, which was latency-bound at
      180 GB/s on a 768 GB/s card (talker -14.4%, CP -22.4% overall for the session).
- [ ] CUDA-12 **Retired: fusing the CP loop onto the device buys nothing.** The plan behind
      CUDA-10 was that the fifteen per-frame host round trips — upload, launch, full sync,
      download — were the cost, and that the cure was a device-resident loop (vLLM-Omni's
      "fuse ~60 kernels"). `qwen_cuda_cp_batch_bench_fused` replays the same fifteen bodies
      with **one** sync and no copies: 11.62 vs 11.53 ms/f at B=4, 13.21 vs 13.17 at B=8.
      **Zero.** The GPU is busy for the whole pass; the host is never the critical path. Do
      not re-open without a measurement that contradicts this one.
- [ ] CUDA-13 **Remaining: `k_matmat_bf16` is still ~2.5x off the memory roof** after the
      unroll, and the talker is now the largest consumer. This is kernel efficiency, not
      structure. The int8/q4 batched matmats share the shape but the CUDA seam is bf16-only,
      so nothing served reaches them.
- [ ] CUDA-14 **Concurrency ladder on the A6000** (0.6B, all three paths, batch=C): RTF p50
      0.44 / 0.77 / 1.05 / 1.37 and stall@250 9% / 53% / 90% / 100% at C2 / C4 / C6 / C8.
      Knee between C4 and C6. Not a qualification — an A6000 behind a 10-core EPYC 7402 is a
      weaker box than the A100 arm, and every run still fails per-class KPI drift.
- [ ] CUDA-11 **Measurement traps. Three now, all the same shape: a harness default that
      quietly disables the thing being measured.**
      **(a)** `tests/serve_soak.py` defaults `--precision` to int8 (`:561`), and the backend
      seam is bf16-only, so a CUDA soak without `--precision default` runs with the GPU at 0%
      while looking healthy.
      **(b)** It also defaults `--prefork-threads` to **1** (`:563`), which with `--prefork 1`
      sizes the whole server pool. Every GPU soak we have run — today's A6000 ladder AND
      yesterday's A100 arm — measured the server with ONE engine thread on a ten-core box.
      Measured at C4: RTF p50 0.68 -> 0.58, stall@1000 11% -> 4%, 82 -> 91 requests, with the CP
      step unchanged at 7.3 ms/frame, so the cost is entirely CPU-side. The ENGINE default is
      `cpus/n` and has always been right; only our measurements were wrong, and every GPU number
      recorded before 2026-09-16 understates the server by about this much. Four threads
      captures it all, eight adds nothing.
      **(c)** Never compare two arms that differ in more than one flag — the
      "seam beats resident" conclusion recorded earlier was really CONVDEC on versus off, and
      had to be withdrawn. Detail: `.work/cuda-parity-track-20260915.md` §12.
- [ ] CUDA-7 **The Metal batched path has the same defect as the CUDA one, unfixed.**
      `qwen_batch_talker_step_ragged` (`qwen_tts_talker.c`) and `batch_cp_transformer_step`
      (`qwen_tts_code_predictor.c`) each have a Metal branch a few lines below the CUDA branch
      that likewise calls `qwen_metal_*_batch_step(...)` without forwarding `active`. The CUDA
      version of this was an illegal memory access and wrong audio (fixed in `c749ac0`); the
      Metal shaders must be read to confirm whether they index by per-slot position the same
      way. NOT fixed here because no Apple GPU was available to verify, and the session was
      scoped to CUDA. Do not assume it is benign.
- [ ] CUDA-3 NEEDS-GPU: `QWEN_CUDA_CONVDEC=1` disables the exact streaming decoder
      (`sd_exact_stream_enabled()` returns 0, `qwen_tts_speech_decoder.c:3062`) and, when not
      streaming, forces `dt_no_overlap = 1` (`qwen_tts.c:1696`), dropping decoder/talker
      overlap. Quantify what streaming actually loses before treating the GPU decoder as a win.
- [ ] CUDA-4 NEEDS-GPU: batched CUDA requires the fused talker AND CP and is capped at `B <= 8`
      (`qwen_tts.c:2964`). Establish whether the cap is a real limit or an arbitrary one.
- [ ] CUDA-5 NEEDS-GPU: re-run the REPRO-1 A/B/B probe against the CUDA server with the fused
      talker on. The fused-GPU delta-prefill guard (`qwen_tts.c:1529-1543`) only forces
      `delta_start = 0` when steering is active, so the no-steering case may still fork.
- [ ] CUDA-6 The backend-agnostic serving layer (admission, execution budget, envelope metrics,
      soak/screen harnesses, KPI contract) should be pointed at the CUDA server unchanged — it
      measures a server, not a CPU, and is the honest way to compare euro for euro. The Arm
      decoder cohort work and specs 11A/12 do NOT transfer: a GPU-resident decoder replaces that
      component rather than tuning it.

- [ ] TQ-7 GPU serving: `--backend cuda --prefork N` is silently broken. **GUARD WRITTEN
      2026-09-15, NOT YET VERIFIED ON A GPU.** `main.c` now refuses the combination up front
      (inside `#if defined(QWEN_HAVE_METAL) || defined(QWEN_HAVE_CUDA)`, and only when
      `gpu_backend_str` is non-NULL, so a CPU-only build does not even compile it and
      `--prefork` without `--backend` is untouched). Message points the user at
      `--batch-size`, which raises throughput inside the single process that owns the context.
      Per-worker GPU contexts (fork first, initialise in each child) remain a possible future
      design, not a bug fix. Still to do on a GPU box: confirm the refusal fires and that
      `--backend cuda` without `--prefork` is unaffected. Original analysis: VERIFIED at HEAD:
      the resident CUDA Talker/CP state is created in `main.c` (~:1665) BEFORE
      `qwen_tts_serve_prefork` (~:3082) forks; a CUDA context does not survive `fork()`, and
      no guard exists anywhere (`grep -ci cuda qwen_tts_server.c` = 0, no mutual exclusion in
      main/qwen_tts/cuda). macOS escapes only via the non-Linux prefork stub. Silent wrong
      answer, not a crash. Fix: refuse the combination, or fall back to the single-process
      batched server with a warning. Related: the global GPU seam is bf16-only
      (`qwen_tts_backend.h` exposes only `matvec_bf16`/`matmat_bf16`), so `--backend cuda`
      with `--int8` offloads nothing while the startup line still advertises offload.
      Scoping note: specs 11A/12 and the Arm decoder work carry NO value on a GPU lane, since
      a GPU-resident decoder replaces that component rather than tuning it; the
      backend-agnostic layers do carry over. Detail:
      `.work/arm-linux-v2-parity-track-20260910.md` section 2e.
- [x] TQ-6 BUILD BREAK, not Arm-specific: the tree does not link when neither
      `__ARM_FEATURE_DOTPROD` nor `__AVX512VNNI__` is defined — `SIMD=portable` (the default
      non-VNNI x86 target) and `SIMD=scalar` both fail. Seven symbols are declared and called
      unconditionally but defined only inside the ISA guard in `qwen_tts_kernels.c`, and the
      `#else` fallback sits inside that guard, so it is unreachable. VERIFIED at HEAD with
      `make blas ARCH_FLAGS="-march=armv8-a"`. Partly introduced by C12-WIN: `_ctx` in
      ddfa5d8, `_pack_stack`/`_stack_epilogue` in edfd3fb. FIXED on `feature/arm-parity-vnni`:
      the ISA-neutral ConvT stack and the DL-4 packer moved outside the guard, no-op fallbacks
      for the three ISA-bound entry points, link-only CI jobs. Re-verified with
      `-march=armv8-a` (links, self-test PASS) and on the native build. Detail:
      `.work/arm-linux-v2-parity-track-20260910.md` section 1.
- [x] TQ-5 HTTP JSON string parsing: **ROOT-CAUSED + FIXED** in
      `cf8dd6b09d6de8abc51cccfa6aa90d3fa062b8c7`. The server now decodes standard JSON
      escapes, UTF-16 surrogate pairs, and raw UTF-8 correctly; malformed strings are
      explicit HTTP 400 errors rather than absent optional fields, and JSON responses
      preserve non-ASCII UTF-8. Causal Turin C1 gate passed: escaped/raw requests converge
      to `tail_len=24`, 53 codec frames, and the CLI-identical full codec SHA. The Python
      harness default `json.dumps()` remains the regression oracle; it was not globally
      changed to `ensure_ascii=False`. Requalification is needed for previous non-ASCII
      semantic-quality/CER/golden evidence. Paired V2/control comparative performance
      evidence remains usable; no full C12 performance rerun is required. Detail:
      `.work/server-cli-italian-correctness-20260910.md`.
- [x] TQ-4 Server-vs-CLI Italian pronunciation defect: **ROOT-CAUSED + FIXED** by TQ-5.
      The defect was upstream JSON decoding, not Talker/CP/KV/V2/GEMM, batching, or the
      decoder. The fixed-tree listening pair is retained privately for human sanity review.
      Previous absolute Italian semantic-quality claims remain pending requalification;
      the existing C12-WIN order resumes unchanged after the Spec12/Spec11A gates.
- [ ] TQ-3 Ear verdict on the paired RES1_V2 bank (`samples/tests/2026-09-09_turin-qualification/`);
      PASS promotes `turin-c8a-32c-vnni-product` from provisional to qualified for C12.
- [ ] Reduce structural decoder intercept/rendezvous cost only where measurements justify it;
      retain fused residual as a qualified pooled candidate and consider a strip executor only for proven
      small-call/intercept work. Ragged worker scratch reuse was rejected as a serving
      optimization; claim-first allocation hygiene is retained but KPI-neutral. Details:
      `.work/p4-rag-panel-scratch-20260907.md`, `.work/p4-rag-claim-first-20260907.md`.
- [ ] No speculative completed-stage resumability or dedicated core lanes without evidence
      (DL-1 is the evidence-gated exception: it is an A/B, not a promotion).

### P5 Ownership and batching

- [x] F3 cross-worker cohort coincidence (2026-09-08): in the cap-2 C4 reference,
      useful natural B>=3 opportunities covered only `2.7%` of steady ready events
      within ±1 ms, `3.6%` within ±2 ms and `11.7%` within ±8 ms. Global batching is
      not justified as the next implementation on this 2x6 host; no state consolidation
      or deliberate batch wait was added. Detail:
      `.work/f3-cross-worker-cohort-coincidence-20260908.md`.
- [ ] EO-1/EO-2 Single-engine/global Talker/CP ready set only after P3/P4 coupling is controlled;
      form deadline-compatible cohorts without waiting solely to create B.
- [ ] AMX Talker/CP only when real B >= 4 work exists. CP stateless re-prefill remains dropped
      unless new local evidence invalidates the reviewed cost model.
- [ ] Later ownership/topology changes only if the bounded overlap evidence justifies them.

### Research-only (not current implementation scope)

- [ ] SL-2 live incremental text / park-not-pad; long-form segmentation with decoder-state
      carry; bounded Talker memory; own-codes re-prompt negative arm.

### P6 Qualification and backend comparison

- [x] QL-1 1.7B final decision on GCP C4 highcpu-16: known-text SL-1 removes the
      dominant long-prefill startup term, but full C3 still lacks sufficient sustained
      tail margin. C2/cap2 is the highest full-envelope GOOD point; C3 is screen-only.
      Detail: `.work/ql1-gcp-c4-highcpu16-17b-final-20260908.md`.
- [x] QL-2a Cross-ISA serving parity audit: common server semantics are portable, but
      AMX Design-D/fused ragged decoder execution is not shared by VNNI or Arm; freeze
      a common-control lane plus a separately labelled best-per-ISA lane before spend.
      Detail: `.work/cross-isa-serving-parity-audit-20260908.md`.
- [x] QL-2b Operational cross-ISA profiles and strict resolved-dispatch gates: AMX,
      VNNI, Arm and common-control profiles pin the relevant flags, reject invalid
      fallbacks and embed the resolved preflight in WAVE/SOAK artifacts. No hardware
      comparison is closed by this task. Detail:
      `.work/cross-isa-operational-parity-20260908.md`.
- [x] QL-2c Local AMD/Turin campaign preparation: known-text SL-1 is pinned across the
  comparable VNNI/Arm/control lanes, the default-off stage-pressure trace has an
  offline receive-gap overlap helper, and the claim audit/runbook preserve
  MEASURED/DERIVED/PREDICTED boundaries. No host was benchmarked. Detail:
  `.work/post-8core-codex-review-20260908.md` and
  `.work/turin-vnni-campaign-plan-20260908.md`.
- [x] QL-2d Turin fast screen on AWS c8a.8xlarge (32 Zen5 cores, 4 CCX, 2026-09-08):
  1.7B holds C8 and not C10 (`2x16` cap 4 STREAM p95 0.79-0.84 at C8, C10 1.05; `4x8`
  cap 2 0.87; `1x32` collapses at 1.4); 0.6B `4x8` cap 4 holds C12 at 250 ms and C16
  at 500 ms, `2x16` collapses at C16. Screen only: provisional profile, 1 wave, short
  texts. Detail: `.work/turin-c8a-32c-fast-screen-20260908.md`.
- [ ] QL-2e Turin ceiling calibration: the doctor's physics ceiling is C28-32 for 1.7B
  where the host delivers 8; measure the three named gaps (wide-pool collapse incl. the
  40 GB/s cross-CCX cache rate, the VNNI decoder term now a ×1.5 GUESS, batch scaling
  past B2) with the stage trace at C8/C10 on `2x16`, then run the pre-registered Phase 4
  on `2x16` cap 4 and `4x8` cap 2 only. Same addendum, §5-6.
- [x] DR-1 Doctor wave plan + ceiling: `wave-plan.json` + `tools/doctor_wave.py`
  (`make doctor-wave`) run the recommended grid from one file; every candidate K gets its
  own measured GEMV roof; section 8 CEILING prints physics / model / floor per shape with
  the measured calibration points of the ISA family. Same addendum, §7.
- [ ] QL-2 Re-evaluate promising backends (0.6B, AVX-512/VNNI hosts, ARM) under the same
  playback-aware harness only after QL-1 has one trusted reference and the QL-2a
  + QL-2b dispatch/quality gates are applied; do not present AMX-only decoder work as
  parity.
  Completed slot: GCP C4 highcpu-16 / 8 physical AMX cores. For 1.7B, `1x8`
  is the best topology and C2 is the final full-envelope point; C3 is screen-only
  and C4 is NOT GOOD. For 0.6B, C3 is the final full-envelope point and C4 is
  non-promoted. The next slot is AMD/Turin VNNI, then Axion/Arm. Detail:
  `.work/ql2-gcp-c4-highcpu16-amx-20260908.md` and
  `.work/gcp-c4-highcpu16-amx-product-capacity-20260908.md`.

### Retained, demoted or deferred (ids kept for addenda; none is a current priority)

- Multi-precision waits behind P0-P3: AMX-2 shared representation, AMX-5 BF16 serving
  policy, AMX-10 W4 feasibility; INT8 is the serving reference.
  PREFILL-Q (calibration-aware prefill quantization) is a deferred research arm behind
  the architecture work; detail: `.work/post-p2-streaming-research-agenda.md` R9.
- Superseded by the envelope: AMX-1, AMX-3, AMX-6, AMX-7, AMX-9 (C4 qualification and
  cross-request decoder aggregation are no longer the next bet; aggregate only for
  isolation/cadence, never for width).
- Controls: P0.1/P0.2, P1.1–P1.4, CTRL-1–CTRL-4; deferred X86-2–X86-8 and LATER-1–4.
- Closed: AMX-4, AMX-5, AMX-8, P3.1, P3.2, P3.6, ragged scheduler review
  (`.work/amx-ragged-scheduler-review-3f7e0df.md`), and the ids below.
- [x] P2.1 Runtime parity — `.work/p2-cross-backend-runtime.md`
- [x] P2.2 CP/Talker region parity — same addendum
- [x] P2.3 Batched-head/budget parity — same addendum
- [x] P2.4 Hot-path allocation fixes — same addendum
- [x] P2.5 One engine-owned budget — same addendum
- [x] P2.6 Pool reentrancy reporting — same addendum
- [x] P3.3a Pool capability parity — `.work/p3-runtime-knob-parity.md`
- [x] P3.4 Decoder capability/policy split — same addendum
- [x] P3.5 Effective AMX/x86 decoder knobs — same addendum

## Qualification gates (provisional, become hard only after MT-1)

| dimension | mandatory | preferred |
|---|---|---|
| correctness | parity PASS; errors = rejects = timeouts = 0 | |
| TTFB / TTFA p95 | measured independently | < 100 ms / < 500 ms (<= 700 ms only for better continuity) |
| STREAM_RTF p95 | < 1 | <= 0.90 (<= 0.85 strong) |
| required_prebuffer p95 | reported | <= 500 ms (<= 250-300 ms strong) |
| safe_play_start p95 | reported | <= ~1 s (<= ~800 ms strong) |
| stall_rate@500ms | -> 0 at the operating point | stall_rate@250ms -> 0 |
| admission / slow client | no induced stall on established streams | |

Never promote q32 for RTF, trade cadence for TTFA, manufacture AMX work, or reopen
BF16/W4 as the P1 fix.

## Evidence

`.work/professional-streaming-architecture.md` (cadence law, AMX accounting, candidates, envelope, historical classification); `.work/p2-checkpoint-20260907.md`, `.work/p1-cadence-truth-20260907.md`,
`.work/p2-sq2-direct-convt-20260907.md`, `.work/p2-sq2-direct-dwconv-20260907.md`, `.work/p2-sq2-direct-input-20260907.md`, `.work/p2-sq2-direct-quant-20260907.md`, `.work/p2-input-length-scaling-20260907.md`, `.work/amx-c4-cross-request-20260907.md`,
`.work/amx-c4-chunk-sweep-20260906.md`, `.work/amx-c4-ragged-threshold-20260906.md`,
`.work/amx-native-epic.md`, `docs/reference-gcp-c4-standard-24.md`, `docs/runtime-map-c8a-c4.md`.
