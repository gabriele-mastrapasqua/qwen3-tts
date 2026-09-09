# C12 architecture review — the engine as it exists after DL-2 + DL-4 (2026-09-09, read-only)

**Task.** Audit, from code and the 2026-09-09 evidence, what limits C12 on the frozen Turin
architecture (`turin-c8a-32c-vnni-product`, HEAD 32f01d4) and rank the 2-5 mechanisms most
likely to buy the next serving win, each with the cheapest experiment that proves it wrong.
**Question.** Is "decoder residency → CP overlap → STREAM tail" the dominant remaining
mechanism at C12, and what is the cheapest path to a sustained STREAM_RTF p95 <= 0.90
including the short/conversational classes?
**Known facts.** Handoff `turin-vnni-final-handoff-20260909.md` §3-§5 (C10-C16 curve, C12
soak 0.912 pooled / short 0.959, C16 soak FAIL, STAGE: CP 22 → 35 ms in overlap, Talker
flat, unit 50.9/53.7 ms, overlap share 30 % in a ramping wave). Three code audits of this
review (frame loop + lane + width; decoder unit; admission/output) are summarised inline
with file:line references; no runtime was changed, no benchmark run.
**Unknowns.** Which CP sections carry the overlap tax (never split per section by overlap);
how much of the sustained short tail is admission stalls vs ramp vs variance; whether the
decoder's f32 weight stream or its activations are the interferer.

## 1. Executive verdict

* **What limits C12 today is not the frame law** (iteration 55-60 ms per 80 ms frame at B3
  ≈ 0.72) but three tails stacked on it in sustained closed loop:
  1. **the CP overlap tax**: +11-14 ms per iteration while a decoder unit is in flight, at
     B >= 2 only (B1: +3.6 ms), and at sustained B3 the lane is busy far more than the
     30 % seen in ramping waves (three 50 ms units back to back per 4 iterations ≈ 60 %);
  2. **inline admission**: every completion in a closed loop admits a new request whose
     prefill runs inside the frame loop and stalls every established stream 60-240 ms
     (qwen_tts.c:3339-3425, helper OFF by default); waves never see this after t=0;
  3. **the short-clip ramp and denominator**: 10 decoder calls for 31 frames (+29 % glue
     on the lane) and a p95 taken over 2.5 s clips, where one stall is +4 % RTF.
* Confidence: the tax and its B-dependence are MEASURED; the width explanation is
  BOUNDED OUT by static-vs-elastic (69.8 vs 69.5 ms); the admission-stall share of the
  short tail is INFERRED (mechanism in code, magnitude measured on the old engine only).
* **<= 0.90 at C12 looks reachable** without a redesign: the tax alone is worth ~0.10 of
  STREAM at sustained B3 (13 ms × 0.6 share / 80 ms), the admission stall a further
  ~0.05 on the short p95. Two mechanisms, both with zero-code first experiments.

## 2. Updated causal graph (batched server, one worker = one CCX)

```
iteration (qwen_tts.c:3288): admission+prefill(inline) → head → sample(per slot) → CP(16 steps)
                              → dec_enqueue(slot at quantum) → Talker → next
decoder lane (1 thread + 4-cpu team): units of different slots run BACK TO BACK (qwen_tts.c:2648-2712)
engine width cap 8→4 from the first enqueue until the queue drains (2748 / 2705)

unit residency (50.9 ms, 4 threads) ── overlap share ──► CP +11-14 ms/iter at B>=2  ──► iteration wall
      ▲                                                   (Talker +0-3)                       │
      │ f32 sgemm weight stream ~330 MB/unit + ~200 MB activation passes                      ▼
      │ (transformer/convnext/init/convt, excluded from int8 by construction)          STREAM_RTF tail
closed-loop admission: inline prefill 60-240 ms per admission ────────────────────────────────┘
short clips: 10 decoder calls/31 frames (ramp 1,2,4 + flush), p95 over 2.5 s
```

Alternatives examined (Mission A):
| mechanism | status | evidence |
|---|---|---|
| pool width 8→4 during a unit | **falsified as the tax** | static 4+4 (always 4-wide) = elastic on Talker+CP within 0.3 ms; per-slot CP sections use B threads either way (code_predictor.c:1264-1370) |
| raw DRAM bandwidth contention on the CCX | plausible, partly contradicted | Talker streams ~1.7 GB/frame at the roof and is untaxed; CP (60 MB int8 re-read 16x/frame) is; a pure bandwidth story would tax both |
| loss of CP's L3 reuse across its 16 steps (eviction by the decoder's 330 MB f32 weight stream + activations) | **best-supported, unmeasured per section** | tax appears only at B>=2 (matmat path with row blocks re-read per step), scales with unit duration, unmoved by activation-side knobs (DL-3) that never touched the weight stream |
| mailbox / handoff | falsified | wait 0.03-0.10 ms/frame, overruns 0 |
| lane wake/yield | falsified | hot-workers arm = no change |
| CP barrier count | unmeasured, constant | 718 barriers/frame regardless of overlap; would show in COST_MAP `LOTHER` |
| request phase alignment | unmeasured | units of 3 slots are staggered by their start iteration; no alignment policy exists |
| admission stalls (closed loop only) | measured on old engine (108-240 ms), not re-measured post-lane | the wave/soak gap of the short class (0.86 vs 0.96) is its signature |
| output socket writes | possible, bounded | synchronous 15 KB writes, no send timeout; healthy clients on the same host: negligible; slow clients: unbounded (PLAN LS-4) |
| stderr serialisation | small, real | `[BATCH] admit/done` are unconditional fprintf across 4 workers |

## 3. Post-V2 bottleneck ranking (one 4-frame unit, 4 threads, ~50 ms)

| rank | cost | ms | kernel today | class |
|---|---|---|---|---|
| 1 | res1 (3 dilated k=7 per block × 4 blocks) | 17.1 | `qwen_conv1d_int8_v2` | compute+bandwidth, already int8 |
| 2 | pre-upsample block: vq, pre-conv 512→1024, in-proj, 8-layer transformer, out-proj, 2 convnext (pw 1024↔4096), initial conv 1024→1536 | ~12 (cost map of 2026-09-09: transformer region 5.2, the rest inside conv_stack 43.3) | **f32 `cblas_sgemm`**, M=4..22, ~255 MB f32 weights per unit ≈ 21 GB/s | **weight-bandwidth bound, f32 by construction** (`qwen_sd_int8_usable` needs in==out<=768, kernels.c:8762) |
| 3 | convt ×4 (k=16/10/8/6) | 8.6 | `causal_conv_transpose1d_blas`: k separate sgemms + f32 panel + scalar scatter (speech_decoder.c:785-812), 75 MB f32 weights | bandwidth + glue |
| 4 | res2 (1x1) | 4.0 | v2 kernel | int8 |
| 5 | resadd + alloc | 5.0 | memcpy of `signal` into `res`, `calloc` of `c2_out`, separate add pass (2377-2437, 2410) | pure glue, ~4 full passes |
| 6 | activation passes around res1: `ext` build, `full` calloc, `cut` copy (2074-2126) | ~3-4 (inside res1's 17.1) | glue, ~95 MB traffic/unit | glue |
| 7 | final conv 96→1 k=7 + clamp | 1.3 | scalar, single-threaded (2267, 2465) | compute |
| 8 | snake ×~13 | 1.0 | AVX2 8-wide poly sin² | compute |
| 9 | ~40 team dispatches, `sd_wq_get_conv` global mutex per conv (511) | <1 | sync | sync |

Exposed by RES1_V2: rows 2, 3, 5, 6 — the f32/BLAS half of the unit that no flag ever
touched (`DIRECT_INPUT`, `STREAM_STRIP`, `FUSED_RESIDUAL` are gated behind AMX Design-D and
are inert on VNNI; only `QWEN_SD_DIRECT_CONVT` is live here).

## 4. Top 5 C12 opportunities

| # | mechanism | expected upside | confidence | impl. cost | correctness risk | cloud cost |
|---|---|---|---|---|---|---|
| 1 | **Prefill off the frame loop** (`QWEN_PREFILL_HELPER=1`, existing path qwen_tts.c:3266; bf16 prefill ~58 ms) | short/conv p95 −0.03..−0.06 in closed loop; pooled −0.01..−0.03 | medium (mechanism certain, magnitude post-lane unknown) | 0 (flag) — then LOW-window tuning | low (existing path, must re-check TTFA and helper vs lane cpu contention) | 2 × 10-min soak |
| 2 | **Decoder pre-upsample + convt in bf16/int8 matmat, one GEMM per convt** ("next RES1_V2") | unit −8..−13 ms (→ ~38-42), overlap share −25 %; if the L3-reuse hypothesis holds, tax per iteration also drops | medium-high on residency, medium on the tax | 2-4 days (bf16 first: `qwen_matmat_bf16` exists) | bf16 low; int8 on the transformer medium (needs the paired bank) | kernel microbench + one B4 A/B + one C12 soak |
| 3 | **Glue removal in the residual unit** (split-input v2 kernel, out-of-place snake, residual in the epilogue, no calloc) | unit −5..−7 ms | high | 1-2 days | low (bit-exact if the tail rows quantise identically) | microbench + B4 A/B |
| 4 | **Short ramp 1,4 instead of 1,2,4** (hard-coded 3536-3541 → flag) | short p95 −0.01..−0.03, lane residency per short clip −1 call | medium | hours | TTFA/prebuffer must not move | short waves + 10-min soak |
| 5 | **Opportunistic B2/B3 units on the lane** (units already queued together; `ng>1` branch exists 2648-2670, forced off in lane mode 2887) | residency per burst −20..−40 % if the f32 weight stream amortises | low-medium (alignment of slots unknown) | 1-2 days after instrumentation | medium (cadence: never wait for a partner) | queue-depth histogram first (free) |

Not in the top 5: phase-aware overlap (section 6), spin/width (section 9).

## 5. "Next RES1_V2" candidates (ranked; file:line in speech_decoder.c unless noted)

1. **bf16/int8 matmat for the pre-upsample block** — transformer linears 2698-2833, convnext
   pw 1819/1841, in/out proj 2652/2960, initial conv 2349, pre-conv 2643. Mechanism today:
   `cblas_sgemm` over f32 weights, M=4 rows → 255 MB streamed per unit. Transformation: pack
   once at load to bf16 (`qwen_matmat_bf16`, kernels.h:186) — halves traffic; int8 VNNI
   quarters it. Affects: the ~12 ms block → ~6 (bf16) / ~4 (int8), AND removes 130-190 MB
   of L3-polluting stream per unit. Risk: bf16 ~free numerically; int8 needs the paired
   bank (transformer, not conv). Falsifier: microbench the block at M=4 on 4 threads; then
   `[STAGE] cp_ms` by overlap at B3 — if the tax per iteration does not move with the
   stream halved, the pollution story is wrong and only residency counts.
2. **ConvT as one GEMM** — `causal_conv_transpose1d_blas` 785: k sgemms (16 for block 0)
   into an f32 panel + scalar scatter. Repack `[in_ch][k·out_ch]`, one sgemm, one scatter;
   then int8. Affects convt 8.6 → ~4-5. Risk: low (same taps, same order per tap).
3. **Residual-unit glue** — split-input for `qwen_conv1d_int8_v2` (kills `ext`/`full`/`cut`
   at 2074-2126), out-of-place snake (kills the `res` memcpy 2377), residual add in the v2
   epilogue (kernels.c:10133-10139), `sd_tmp_alloc` instead of calloc (2410, 2119).
   Affects resadd+alloc 5.0 → ~1, res1 −3. Risk: low.
4. **Vectorised, threaded final conv + clamp** 2267/2465: 1.3 → ~0.2 ms. Risk: nil.
5. **Lock-free `sd_wq_get_conv`** 511 and arena scan 92: sub-ms, matters only with
   several lanes decoding at once (they do: 4 workers).

Together 1-4 plausibly take the unit from ~51 to ~35-38 ms on 4 threads, i.e. the target
that DL-4 missed by 2-5 ms, with margin.

## 6. Phase-aware overlap verdict: NO-GO as a scheduling policy at C12, YES as unit length

Order of record: CP → enqueue → Talker (qwen_tts.c:3516-3768). A 50 ms unit started after
CP covers this iteration's Talker (32-35 ms) and the next iteration's head/sample and the
first ~10 ms of CP — the policy already puts the unit on Talker first. At sustained B3 the
lane runs three units back to back (~150 ms per ~240 ms of loop), so 60 % of every phase is
overlapped whatever the start policy; confining units to Talker windows would need units
<= 30 ms or gaps in the lane that do not exist at B3. Quantified: tax 11-14 ms × share
0.6 ≈ +8 ms/iteration ≈ +0.10 STREAM — the whole preferred gap; a start-delay policy
could move at most the B1-B2 fraction. Sub-frame units were already measured worse.
Cheapest falsifier of the whole direction (zero code): `[STAGE]` at C12 with B3 pinned
(`--batch-cap 3`, long bank): if overlap share is >= 55 %, phase placement cannot buy the
gap; only shortening the unit (section 5) can. Keep C12-WIN-2 as a paper check only.

## 7. Short-tail verdict

Contributions (2.5 s clip, 31 frames, `_BUSY=4`):
* ramp 1,2,4 → 10 decoder calls vs 7.75 (+29 % lane glue; each call carries ~40 team
  dispatches, warm-tail materialisation, allocs) — adds lane residency, i.e. overlap;
* closed-loop admission: inline prefill at the top of the iteration stalls every stream;
  at C12 with short clips ≈ 1 admission/s/worker; TTFA excluded but the stall lands inside
  other streams' STREAM; the wave/soak gap of the short class (0.862 → 0.959 p95, p50 only
  0.813 → 0.860) is its signature;
* variance: p95 over 2-3 s clips amplifies any 60-100 ms stall (+3-4 % RTF each);
  medium/long average them out (their p50/p95 spread is 0.05, short's is 0.10).
Smallest discriminating experiment (zero code, one 10-min C12 soak each):
(a) `QWEN_PREFILL_HELPER=1` — if the short p95 drops >= 0.03 with TTFA p95 unchanged,
admission is the tail; (b) `[STAGE] admit_ms/prefill_ms` histogram on the control soak
(STAGE trace on one worker) — the direct measurement of (a)'s magnitude; (c) ramp 1,4
(flag, Codex) only after (a)/(b). Do not use q8.

## 8. Old-experiment reassessment

| idea | old result | old falsifier | what changed | falsifier valid | retest | cheapest retest | upside |
|---|---|---|---|---|---|---|---|
| 2+6 / 5+3 / 6+2 splits | worse | per-slot sections on <4 step threads | nothing on the step side | YES | NO | — | — |
| static 4+4 vs elastic | equal Talker+CP | width not the tax | nothing | YES | NO | — | keep elastic |
| direct ConvT (`QWEN_SD_DIRECT_CONVT`) | no tax change (DL-3) | CP-in-overlap ms | still the only live direct flag; targets a 75 MB f32 stream + panel | PARTIAL: it only removed the `full_len` buffer, not the k sgemms | NO as-is; YES as "one GEMM + int8" (section 5.2) | microbench | 4-5 ms/unit |
| direct dwconv / direct input / strip / fused residual | neutral | tax | **inert on VNNI** (gated behind AMX-D) — the old "neutral" was a no-op | NO (never ran here) | YES, as the VNNI glue work of section 5.3, not as flags | microbench | 5-7 ms/unit |
| sub-quantum units (SUBQ) | worse | lane saturation | nothing | YES | NO | — | — |
| q8 units | cadence FAIL | prebuffer 806 ms | nothing | YES | NO | — | — |
| NTA prefetch | neutral | tax | applied to panel-kernel weights only; the 330 MB sgemm stream never had it | PARTIAL | only inside a custom decoder matmat (NT loads) | part of 5.1 | pollution |
| hot lane workers / panel sizing | neutral | unit time | nothing | YES | NO | — | — |
| inline decoder batching (`QWEN_DECODER_BATCH`) | stall bug / off the lane | critical path | lane exists; `ng>1` branch is reachable off-loop | PARTIAL | YES as opportunistic lane grouping (section 10) | queue-depth histogram | 20-40 % residency |
| CP int4 (fits L3) | quality collapse on late codebooks | quant ladder | nothing | YES | NO | — | — |
| VNNI prepack | +1-3 % worse | Talker/CP ms | nothing | YES | NO | — | — |
| 1x32 / 2x16 | thrash | STREAM | nothing | YES | NO | — | — |
| pool spin 4096 vs 65536 | 0.893 → 0.808 on 2x16 C8 | STREAM | 4x8 screen ran at 4096 and read 0.80-0.81 at C12 vs 0.82-0.85 frozen; confounded with profile/prefill/MR | PARTIAL | YES, one short screen (C12-WIN-5A) | 2 × short wave | 0-0.03 |
| prefill helper (LS-4) | deferred P3 | inline stall 108-240 ms | lane freed the loop; helper cost now competes with the decoder lane, not with inline decode | PARTIAL | **YES, first** | flag + 10-min soak | short p95 −0.03..−0.06 |
| AMX Design-D / ragged AMX decoder | AMX hosts only | — | no AMX on Turin | n/a | NO here | — | — |

## 9. Width / bandwidth geometry

* CP and Talker row-split sections are int8 GEMV-shaped and saturate the CCX at 2-4
  threads; per-slot sections use exactly B threads whatever the width (RMINE, code_predictor.c
  1274-1348). Width 4 during a unit therefore costs < 1 ms (static-vs-elastic). An
  asymmetric Talker-8 / CP-4 policy has no mechanism to help: CP is not width-bound and
  Talker is not taxed. **Drop the phase-width part of C12-WIN-5**; keep the spin screen
  only because it is confounded with the 0.80 → 0.83 screen-vs-frozen difference.
* The geometry that matters is the decoder's: 4 threads streaming 330 MB f32 weights and
  ~200 MB of activation passes per 50 ms through the same 32 MiB L3 and CCX memory path
  as CP's 16× re-read of 60 MB. Halving that stream (bf16) is the cheapest geometry change.
* Diagnostic that settles width vs pollution without code: `QWEN_COST_MAP=1` per-section
  CP accumulators bucketed by the `[STAGE] overlap` flag (QKV/WO/GATEUP/DOWN/LMHEAD inflate
  only ⇒ width; ATTN/LOTHER inflate too ⇒ pollution).

## 10. Decoder B2 on the lane: MEASURE (instrumentation first), not GO

Units of 3 slots queue on one FIFO and run sequentially; the `ng>1` grouped call exists
but is forced off in lane mode. If two units are already queued when the worker wakes,
grouping them is free of waiting and amortises the ~330 MB f32 weight stream and the ~40
dispatches; if they are not queued together, waiting for a partner costs lead and is
forbidden. Unknown: how often units coincide (slots' quantum boundaries are staggered by
their admission iteration). Cheapest step: count queue depth at each worker wake
(`dp->queued` histogram in `[serve-profile]`, one counter). GO only if depth >= 2 is common
(>= 30 % of wakes at B3) and the grouped path is per-item-exact; reject on any prebuffer
or safe-start move. After section 5 shrinks the unit, the coincidence rate drops — do 5 first.

## 11. Immediate experiment ladder (cheapest discriminating first; all vs the frozen control)

1. **Zero code, one soak**: `QWEN_PREFILL_HELPER=1`, C12, 10 min, short+conversational p95,
   TTFA p95, `[STAGE]` on one worker. Decides whether admission is the short tail.
2. **Zero code, one diag wave**: `QWEN_COST_MAP=1` + `QWEN_STAGE_TRACE=1` at `--batch-cap 3`
   long bank: CP sections by overlap flag; overlap share at pinned B3. Decides width vs
   pollution and kills/keeps phase-aware.
3. **Zero code, short screen**: `QWEN_POOL_SPIN` 4096 / 16384 vs 65536, C12 short+mixed waves.
4. **Kernel-level, no serving**: microbench of the pre-upsample block at M=4/4 threads in
   f32 vs bf16 (`qwen_matmat_bf16`), and convt as one GEMM. Go/no-go on section 5.1/5.2.
5. **Glue removal** (5.3/5.4) with the v2 self-test extended; B4 single-CCX A/B on unit
   time and CP-in-overlap.
6. **B4 A/B then C12 soak** of the combined decoder change; promotion by the C12-WIN-7 gate.
7. Ramp 1,4 flag (only if 1 says admission is not the whole short tail).
8. Queue-depth histogram → decide section 10.

## 12. Architecture after C12

* **Incremental / resumable prefill** stays the major post-C12 item (Mission I): admission
  is both the TTFA-vs-length problem and, in closed loop, a STREAM tail; a bounded initial
  text window with resumable KV is the same mechanism solving both. Order: helper first
  (cheap), then incremental prefill as its own P-track; do not fold it into C12-WIN.
* **Lead-aware scheduling**: the lead gate exists (`QWEN_STREAM_LEAD_GATE`); use it as a
  fairness tool once the tax is gone, not as a tax workaround.
* **Non-blocking output**: synchronous 15 KB writes with no send timeout are correct for
  healthy clients and unbounded for slow ones (LS-4); the async writer exists behind
  `QWEN_SERVER_ASYNC_OUTPUT=1` and needs its own qualification, not a C12 detour.
* **Long-form continuity**: the exact-streaming decoder already carries per-layer state;
  the missing piece is Talker-side (resumable text), same item as incremental prefill.
* **Batching**: stage-local only — the lane grouping of section 10; no global batching.
* **AMX/VNNI evolution**: the VNNI decoder work of section 5 is ISA-generic in structure
  (bf16/int8 matmat + glue removal); on AMX hosts Design-D already covers part of it; keep
  one decoder dataflow with two leaf families rather than two decoders.
* **Fail-fast boundary** (TQ-2): the parent writes the 503 and closes without reading the
  body (qwen_tts_server.c:3031-3035) → RST/EPIPE; fix = `shutdown(SHUT_WR)` + bounded
  drain before `close`. Also: health/speakers requests occupy a dispatch slot until close.

## 13. DO-NOT-DO

* No sub-frame decoder units, no q8, no 2+6/5+3/6+2, no 1x32/2x16, no CP int4, no
  VNNI prepack, no static split — all remeasured or unchanged since their falsifiers.
* No phase-aware unit scheduler before experiment 2 shows overlap share < 50 % at B3.
* No asymmetric Talker/CP widths.
* No inline decoder batching; no waiting for a batching partner on the lane.
* No 30-min soak for a candidate that has not moved a B4 single-CCX A/B beyond noise.
* No int8 on the decoder transformer without the paired bank and the ear check.
* No new profiler framework: `[STAGE]`, `QWEN_COST_MAP`, `[SDUP]` already answer the
  questions above.

## 14. What would falsify this direction

* Experiment 1 moves the short p95 by < 0.01 and `admit_ms` shows < 30 ms per admission:
  admission is not the short tail → the tail is ramp + variance; go to ramp 1,4 and accept
  a class boundary rather than a class gate.
* Experiment 2 shows the CP tax entirely in row-split sections: width is the mechanism
  after all → the lever becomes "no narrowing during CP" (one-line guard at 2748) and the
  decoder must tolerate sharing its cpus — a different track.
* Section 5.1 halves the decoder's weight stream but CP-in-overlap stays at 35 ms: the
  interferer is the activation traffic, not the weights → the remaining lever is residency
  only, and <= 0.90 at C12 then depends on the unit reaching ~35 ms.
* Overlap share at pinned B3 is < 40 %: the tax is a smaller share of the gap than
  estimated and the admission/ramp terms dominate.
