# AMD Turin c8a.8xlarge (32 cores) — fast screen and ceiling (2026-09-08)

Status · OPEN. First host of the Turin/VNNI campaign; this is a FAST SCREEN, not the
pre-registered Phase 4 of `.work/turin-vnni-campaign-plan-20260908.md` (see §2).

Task · QL-2d (screen, done) and QL-2e (ceiling calibration, open) in `PLAN.md` P6;
tooling DR-1 (doctor wave plan + ceiling) in the same section.

Question · How many realtime streams does a 32-core Zen5 host hold for 1.7B and 0.6B,
which topology holds them, and can the doctor say the theoretical maximum before a
wave is paid for?

Verdict · 1.7B: **C8 is the practical ceiling** on this host. Every shape sits at
`TOTAL_RTF` p95 0.91-1.00 at C8; C10 crosses 1.0. Best C8 shape is `2x16` cap 4
(`STREAM_RTF` p95 0.79-0.84, stall@250 0 %); `4x8` cap 2 with `QWEN_STREAM_DECODE_CHUNK=4`
`QWEN_DECODER_BATCH=1` has the lowest required prebuffer (188-228 ms). No shape holds C8
at a 100 ms client buffer. `1x32` is dead (STREAM 1.3-1.5, prebuffer > 1 s). 0.6B: `4x8`
cap 4 holds **C12** at 250 ms (STREAM p95 0.72) and C16 at 500 ms (0.83); the wide `2x16`
pool collapses at C16 (1.22). The doctor's physics ceiling (bandwidth only, perfect
batching, free decoder) is C28-32 for 1.7B: the gap to the measured C8 is decoder plus
glue plus the wide-pool collapse, not the weight stream.

## 1. Ground truth

| item | value |
|---|---|
| branch / source | `feature/x86-amx-vnni-oss`, `6497808:clean` |
| host | AWS `c8a.8xlarge` spot, us-east-1; AMD EPYC 9R45 (Zen5) |
| topology | 32 physical cores, SMT off, 1 socket/NUMA, 4 LLC domains × 8 cores (L3 32 MiB each), 61 GiB |
| OS / toolchain | Linux 7.0.0-aws, gcc 15.2, distro OpenBLAS |
| build | `make blas` → auto `SIMD=avx512bf16`; isa_class `x86_avx512bf16`; no AMX |
| gates | SMT off PASS · governor PASS · cgroup PASS · `--self-test` PASS · dispatch gate PASS · `make cpu-check` PASS 18 / FAIL 0 |
| regions | `[talker] batched step as one parallel region: ON (team 16)`, `[cp] transformer step … ON`, `[cp] whole decode frame … ON (BW 4)`: Talker and CP already run batched per worker |

GEMV roofs (`tests/roof_matvec_int8.c`, the Talker's own int8 kernel, MEASURED):

| mask | threads | DRAM GB/s | 101 MB cache-resident GB/s |
|---|---|---|---|
| 0-3 | 4 | 55.2 | 70.4 |
| 0-7 (one CCX) | 8 | 55.0 | 65.7 |
| 0-15 (two CCX) | 16 | 109.9 | **40.5** |
| 0-31 (host) | 32 | 208.3 | 333.8 |

One CCX gets 55 GB/s whatever the thread count; the host roof is the sum of four. The
cache-resident rate on the 16-thread mask is BELOW its DRAM rate: a set split across two
CCX thrashes between them, which is the first physical reason wide pools lose.

## 2. Protocol and its deviations from the pre-registered plan

`tests/serve_parallel_wave.py`, int8, `ryan`, seed 42, `tests/load_texts_en.txt`
class `short` (5 texts, ~2.1-2.3 s audio, 7 chunks), synchronized waves, all C fired at
t=0. Deviations, deliberate to get a first look in one evening:

* profile `aws-c8a-16c-vnni-ttfa` (provisional), not `vnni-product`; it sets
  `QWEN_STREAM_DECODE_CHUNK=8` and `QWEN_DECODER_BATCH=0` where the doctor draft pins 4 / 1;
* 1 wave per level (the plan says 3), `short` only (the plan says short/medium/long/mixed),
  `--no-crosscheck`, explicit `--batch-cap` per shape, no `--language`.

So every number below is a SCREEN. It ranks shapes and finds the cliff; it does not
qualify an operating point. Logs: `~/bench/step1*.log`, `~/bench/step_all.log`,
`~/bench/beyond-c8/` on the host (untracked evidence).

## 3. Results — 1.7B

Cap = in-flight requests per worker. Levels listed in run order; a repeated C is the same
server warm.

| shape | C | TTFA p95 ms | STREAM p95 | TOTAL p95 | prebuffer p95 ms | stall @100/@250/@500 |
|---|---|---|---|---|---|---|
| 4x8 cap2 | 4 | 106 | 0.71 | 0.75 | 67 | 0 / 0 / 0 % |
| 4x8 cap2 | 6 | 173 | 0.87 | 0.95 | 327 | 67 / 0 / 0 % |
| 4x8 cap2 | 8 | 181 | 0.87 | 0.95 | 334 | 100 / 12 / 0 % |
| 4x8 cap2 warm ×3 | 8, 8, 8 | 187 / 174 / 172 | 0.92 / 0.87 / 0.90 | 1.00 / 0.94 / 0.97 | 357 / 295 / 344 | @250: 25 / 0 / 12 % |
| 4x8 cap2 + chunk4 + decoder batch 1 | 8, 8 | 180 / 176 | 0.92 / 0.89 | 0.99 / 0.97 | 228 / 188 | @250: 0 / 0 % |
| 2x16 cap4 | 4 | 115 | 0.53 | 0.58 | 29 | 0 / 0 / 0 % |
| 2x16 cap4 | 8, 8 | 223 / 210 | 0.84 / 0.79 | 0.94 / 0.91 | 276 / 189 | @250: 0 / 0 % |
| 2x16 cap5 + chunk4 + decoder batch 1 | 8 | 224 | 0.90 | 1.00 | 237 | @250: 0 % |
| 2x16 cap5 + chunk4 + decoder batch 1 | 10 | 270 | **1.05** | 1.17 | 346 | @250: 0 % (short texts only) |
| 1x32 cap8 | 4 | 246 | 0.67 | 0.79 | 124 | 0 / 0 / 0 % |
| 1x32 cap8 | 8, 8 | 418 / 386 | 1.55 / 1.40 | 1.73 / 1.56 | 1287 / 1084 | @500: 62 / 62 % |

Warm vs cold moves TTFA by 10-15 ms and nothing else. Chunk 4 + decoder batch 1 lowers the
required prebuffer by ~100 ms at equal RTF. Zero errors, rejects or timeouts anywhere.

## 4. Results — 0.6B

| shape | C | TTFA p95 ms | STREAM p95 | TOTAL p95 | prebuffer p95 ms | stall @100/@250/@500 |
|---|---|---|---|---|---|---|
| 4x8 cap2 | 4 / 6 / 8 | 60 / 98 / 101 | 0.45 / 0.60 / 0.59 | 0.45 / 0.61 / 0.60 | 11 / 40 / 42 | 0 % everywhere |
| 4x8 cap4 | 8 | 103 | 0.63 | 0.64 | 73 | 0 / 0 / 0 % |
| 4x8 cap4 | 12 | 133 | 0.72 | 0.74 | 155 | 17 / 0 / 0 % |
| 4x8 cap4 | 16 | 168 | 0.83 | 0.86 | 373 | 100 / 25 / 0 % |
| 2x16 cap8 | 8 | 133 | 0.68 | 0.69 | 132 | 0 / 0 / 0 % |
| 2x16 cap8 | 12 | 186 | 0.88 | 0.93 | 485 | 100 / 58 / 0 % |
| 2x16 cap8 | 16 | 253 | **1.22** | 1.28 | 1129 | 100 / 100 / 94 % |

## 5. Ceiling — what the doctor now says, and how far to trust it

`tools/doctor.py` section 8 (added with this task) gives three numbers per shape from the
measured per-mask GEMV roofs: **physics** (Talker + CP bytes at the roof, perfect batching
at B2/B1 = 1.10 per extra slot, decoder free), **model** (plus the decoder term, rho ≤ 1.0),
**floor** (W × 1, the shape if the effective per-worker batch stays ~1).

| model | shape | roof GB/s | physics C (B) | model C (B) | floor | measured cliff |
|---|---|---|---|---|---|---|
| 1.7B | 1x32 | 208 | 16 (16) | 16 (16) | 1 | C8 already 1.40 — FALSIFIED |
| 1.7B | 2x16 | 110 | 32 (16) | 12 (6) | 2 | C8 0.84, C10 1.05 — model optimistic |
| 1.7B | 4x8 | 55 | 28 (7) | 4 (1) | 4 | C8 0.87 — model pessimistic |
| 0.6B | 2x16 | 110 | 32 (16) | 16 (8) | 2 | C12 0.88, C16 1.22 |
| 0.6B | 4x8 | 55 | 64 (16) | 12 (3) | 4 | C12 0.72, C16 0.83 |

Reading. The weight stream is not the wall on this host: physics says 28-32 streams of 1.7B
and the machine delivers 8. The gap has three named parts, in order of evidence:

1. **Wide-pool collapse** (MEASURED): 1x32 and 0.6B 2x16 fall off a cliff the model does
   not contain. The 16-thread cache-resident rate of 40 GB/s (§1) is the physical half; the
   other half is the per-step rendezvous of a 16-32 thread team on 4 CCX.
2. **Decoder term** (GUESS): on a non-AMX ISA the doctor charges the Design-D calibration
   × 1.5. At the model ceiling the decoder is 51-64 % of the frame on 2x16 and 28 % on 4x8.
   It has never been measured on VNNI; it is the term that decides whether 2x16 can go past
   C8.
3. **Batch scaling** (TRANSFERRED): B2/B1 = 1.10 was measured at B2 on the AMX reference;
   the physics number extrapolates it to B7-B16, which nobody has measured.

The calibration points of §3-4 are now in `tools/doctor.py` (`CAL_POINTS`) and print
under section 8 for the `x86_avx512bf16` family, so the next VNNI box starts with the
model's trust boundary on the page.

## 6. Unknowns → QL-2e

* Frame decomposition at C8 and C10 on 2x16 with `QWEN_STAGE_TRACE=1` (Phase 6 of the
  campaign plan): decoder vs Talker/CP vs serial, so the ×1.5 becomes a measured VNNI
  decoder term (`dec_glue_ms`, `dec_per_item_frame_ms` for this ISA).
* Effective per-worker B per step (the `B` column of the wave is in-flight, not per step):
  the cost-map/census can say whether the B=4 regions actually ran at B=4 on short texts.
* The 16-thread cross-CCX CP rate: is the 40 GB/s cache-resident reading the CP's real
  rate on 2x16, and does a CCX-aware row split recover the 65 GB/s of one CCX?
* Pre-registered Phase 4 (`vnni-product`, 3 waves, short/medium/long/mixed) on the two
  surviving shapes only: `2x16` cap 4 and `4x8` cap 2 (+ chunk 4 / decoder batch 1).

## 7. Tooling shipped with this task (DR-1)

* `tools/doctor.py`: `wave_plan()` writes `wave-plan.json` next to `doctor.txt` — the
  recommended and alternative shapes at their predicted cap, C as `[C, C, C+2]` (cold, warm,
  one past), for 1.7B and 0.6B, plus ISA-filtered A/B candidates; every candidate K now gets
  its own measured GEMV roof (scaling 8T → 16T by the membw sweep predicted 198 GB/s where
  the mask measures 110); section 8 CEILING with physics / model / floor and the measured
  calibration points of the ISA family.
* `tools/doctor_wave.py` (`make doctor-wave`, `WAVE_ARGS="--dry-run"`, `--only <label>`):
  runs the plan sequentially from one file, one log per run, summary table, exit code = failed
  runs. No shell chain, no process polling.
* `tests/test_doctor.py` covers the plan, the runner's dry run, the ceilings and the
  calibration rendering.

What changed: PLAN P6 gains QL-2d (done), QL-2e (open) and DR-1 (done), all pointing here.

## 8. Tier-1 results (2026-09-09, from `.work/32c-serving-utilization-architecture-review-20260909.md` §15/§18.20)

All MEASURED on the same host and binary, one wave each, profile `aws-c8a-16c-vnni-ttfa`
unless stated, logs `~/bench/t1/` (untracked). "Lane" = one worker pinned with `--cpu-mask`.

**Undiluted lane law (one fixed short text, `1x8@0-7`, `QWEN_STAGE_TRACE=1`)** — per iteration
by true active B, ms: 

| model | B | Talker | CP | decoder (amortized) | head+sample+output+serial | wall mean | wall p95 | STREAM p95 |
|---|---|---|---|---|---|---|---|---|
| 1.7B | 1 | 26.2 | 18.1 | 7.4 | 0.25 | 52.3 | 81-88 | 0.674 |
| 1.7B | 2 | 28.4 | 20.8 | 15.1 | 0.27 | 64.5 | 128 | 0.861 |
| 1.7B | 3 | 27.9 | 22.3 | 23.2 | 0.28 | 73.7 | 162 | 0.991 |
| 1.7B | 4 | 31.1 | 23.6 | 31.3 | 0.31 | 86.2 | 205 | 1.172 |
| 0.6B | 1 | 8.7 | 17.2 | 7.1 | 0.13 | 33.1 | 63 | 0.433 |
| 0.6B | 4 | 11.1 | 22.8 | 28.9 | 0.17 | 63.0 | 195 | 0.889 |

Lane law: 1.7B `T(B) ≈ 40 + 13.5·B` ms; 0.6B `≈ 23 + 12.3·B`. Per slot: decoder 7.8 amortized
(9.7 per decoded slot-frame, 72 %), Talker +1.6, CP +1.8; the loop-thread serial tax is
~0.05 ms per slot. The wall p95 rows are the iterations in which the slots' 8-frame decoder
calls coincide (synchronized cadence): 2.5× the frame budget at B4 — this, not the mean, sets
the required prebuffer.

**Decoder call anatomy (`QWEN_SD_PHASE=1`, 8T, 8-frame call, both models identical):** 59-63 ms
= 7.4-7.9 ms/frame; `conv_up` 88-92 % of the call; inside it `res1` (dilated k=7 int8 VNNI
conv1) 28-30 ms = 48 %, `convt` 8.0-8.5, `res2` 7.2, `resadd` 3.2-3.9, `alloc` 1.8-2.7,
`final` 2.5, `snake` 1.0. The ramp calls (1 and 2 frames) cost 17-19 and 15 ms/frame.
Thread scaling of the decoder per frame: 1 core 33 ms (0.6B), 2 cores 18.5 (1.7B), 4 cores
12.9, 8 cores 7.6-9.7. One core keeps one stream's decoder realtime with 2.4× margin.

**T1-a, two 4-core lanes on one CCX (1x4@0-3 + 1x4@4-7, 1.7B, C2 each, concurrent):** STREAM
p95 1.278 / 1.286 vs 0.972 solo; Talker 63.7 ms/step (27.8 solo), CP 55.7 (23.2), decoder 32.9
(24.3). Two weight streams on one CCX halve each other: **a CCX holds exactly one step-lane**.
The 2-thread lane (`1x2@0-1`, 1.7B B1) costs Talker 26.4 + CP 20.8 ms = the 8-thread cost, so
the weight stream saturates the CCX at 2 threads and the other 6 cores are free for work that
does not stream weights. The partition question is therefore not bandwidth but whether the
decoder's f32 activations evict the CP's L3-resident rows (CP is 18 ms with residency vs 33 ms
pure DRAM): UNKNOWN, falsifiable with `tests/decode_quantum_bench` pinned on cores 2-7 while
a `1x2@0-1` lane runs.

**T1-c, spin budget on the wide shape (2x16 cap 4, C8, 1.7B):** `QWEN_POOL_SPIN` 4096 → STREAM
p95 0.893, TOTAL 0.988, csw 38k/s; **65536 → 0.808 / 0.910, csw 7.6k/s**, cores 26.9; 256 →
0.999 / 1.127, csw 197k/s. Part of the 16-thread width tax is park/wake and is bought back by
a larger spin budget (the opposite of the old c8a-16c soak verdict; needs a SOAK before
promotion). Note the run-to-run spread of the 2x16 C8 point: 0.79-0.89 across five one-wave
screens.

**T1-e, lane B3 on long texts, SL-1 pinned (`QWEN_TTS_STREAM_LAYOUT=1`, `1x8@0-7`):** B3 long
0.917 (TOTAL 0.923, TTFA p95 239), B3 short+medium+long 0.965 (1.046), B2 long 0.792 (0.796).
Long texts make the lane slightly better (start-up ramp and prefill amortize); the short-bank
law is not optimistic.

What changed: this section; `tools/doctor.py` `CAL_POINTS` carries the lane law and the
two-lane result; the review addendum gained §19.

## 9. Pre-registered lane, first 3-wave cells (2026-09-09, §18.20 step 6)

`--profile vnni-product` (status unqualified; pins `QWEN_TTS_STREAM_LAYOUT=1`,
`QWEN_DECODER_BATCH=1`, `QWEN_PREFILL_MATMAT=0`, `QWEN_NO_BF16DOT=1`, `QWEN_NO_BF16_MATMUL=1`),
`--language English`, classes short+medium, 3 waves, seed 42, logs `~/bench/step6/`.

| shape | C | TTFB p95 | TTFA p95 | STREAM p50/p95 | TOTAL p95 | prebuffer p95 | safe-start p95 | max gap p95 | stall @100/@250/@500 | cores | csw/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2x16 cap4 | 4 | 626 | **1288** | .514/.555 | 1.287 | 52 | 1339 | 173 | 0/0/0 % | 16.8 | 26.8k |
| 2x16 cap4 | 8 | 1867 | **2566** | .784/.900 | 2.456 | 308 | 2874 | 360 | 83/0/0 % | 15.5 | 26.1k |
| 4x8 cap2 | 4 | 2 | **686** | .686/.714 | 1.081 | 61 | 744 | 223 | 0/0/0 % | 16.7 | 5.4k |
| 4x8 cap2 | 8 | 648 | **1341** | .834/.885 | 1.689 | 217 | 1557 | 280 | 100/0/0 % | 15.7 | 6.9k |

Reading (MEASURED): the steady-state stream is as good as or better than the provisional
profile (4x8 C8 STREAM p95 0.885, 2x16 C8 0.900, both stall@250 0 %), but **first audio is
10× later** (686-2566 ms vs 106-223) and TOTAL_RTF is 1.1-2.5. The cause is the lane's own
policy: `QWEN_PREFILL_MATMAT=0` + `QWEN_NO_BF16*=1` route the prefill to the f32 BLAS
fallback ("this lane intentionally does not claim native AVX-512 BF16 prefill"), which on
this host costs ~600 ms per admission instead of ~58 ms, serialized on each worker's loop
thread — hence TTFB of 0.6-1.9 s at C8 (the header waits behind queued prefills) and half
the machine idle (cores 15-17 of 32). Zen5 has native `avx512_bf16`; the VNNI-only policy
was written for hosts without it. **Do not run Phase 4 on Turin with `vnni-product` as
committed**: derive a `vnni-bf16-product` lane (same pins, `QWEN_PREFILL_MATMAT=1`, bf16 dot
allowed where the CPU reports `avx512_bf16`) or qualify the provisional profile, then
re-run these cells. The step-6 cells stand as a control: STREAM numbers of the two lanes
agree within 0.02-0.05; the difference is startup only.

## 10. L3-contention falsifier and the intra-CCX pipeline verdict (2026-09-09)

Pre-registered (review §19.3): a 2-core step lane `1x2@0-1` (1.7B, one fixed short text,
`QWEN_STAGE_TRACE=1`) alone and with a decoder-only load pinned on cores 2-7
(`qwen_tts_decode_quantum qwen3-tts-1.7b 6` in a loop, 6 threads, ~28 ms/frame of decoder
work per call, i.e. a heavier and more continuous load than the real per-slot decoder). Kill
signature: CP per iteration 18-21 → ~33 ms (residency lost). Logs `~/bench/l3/`.

| arm | active | CP ms | Talker ms | decoder (amortized) | wall mean | wall p95 | STREAM p95 |
|---|---|---|---|---|---|---|---|
| control | 1 | 20.8 | 26.5 | 16.8 | 64.3 | 132 | 0.868 |
| control | 2 | 31.0 | 33.3 | 33.9 | 98.5 | 233 | 1.344 |
| loaded | 1 | 23.6 (+13 %) | 29.3 (+11 %) | 17.3 | 70.4 (+9.5 %) | 144 | 0.948 |
| loaded | 2 | 33.8 (+9 %) | 35.6 (+7 %) | 34.4 | 104.1 (+6 %) | 247 | 1.419 |

**Falsifier result: PASS.** The kill signature did not occur: the CP keeps its L3 residency
(21 → 24 ms, not 33) and the whole step lane pays +12 % (47.3 → 52.9 ms at B1) for a decoder
load heavier than the pipeline would carry. Bandwidth and L3 sharing inside one CCX between a
step lane and a decoder lane is a bounded, measured tax, not a wall.

**But the pre-registered split is wrong on the step side (MEASURED).** On 2 threads the per-slot
sections of the regions (norms, RoPE, attention, SwiGLU, quant, argmax: one slot per thread
between barriers) cost +17 ms per slot (Talker +6.8, CP +10.1 from B1 to B2), against +6.4 on
4 threads and +3.4 on 8. A 2-core step lane reaches B4 at ~98 ms before any decoder: NO-GO.

Pipeline arithmetic with measured terms (per lane, 80 ms budget; contention +12 % applied
to the step lane):

| split | T_step(B) | decoder lane | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|
| 2 step + 6 decoder | 47 + 17·(B−1) → ×1.12 | 6 cores: 1 stream per core at 33 ms/frame, parallel | 91 ms (1.14) | 110 (1.4) | — | — |
| **4 step + 4 decoder** | 44.6 + 6.4·(B−1) → ×1.12 | 4 cores: 12.9 ms/slot-frame serial, or 1 core per stream at 33 ms parallel | max(64, 39) = **64 (0.80)** | max(72, 52) = **72 (0.90)** | max(79, 65) = 79 (0.99) | 86 (1.08) |
| today, 8 threads inline | 40 + 13.5·B | — | 79 (0.99) | 94 (1.17) | — | — |

**Verdict: GO for the intra-CCX decoder/step pipeline in its 4 + 4 form; NO-GO for 2 + 6.**
Expected gain per lane: B3 from 0.99 to ~0.80 (GOOD), B4 from 1.17 to ~0.90 (gate edge),
B5 at the hard gate. On four lanes: C12 GOOD and C16 at the preferred-gate edge, from C8
today. TTFA unchanged (the first chunk overlaps nothing). Moving the decoder to *another*
CCX is not needed and is worse on the evidence: it would carry the decoder's activations
across L3s (the 40.5 GB/s two-CCX signature) and cost the destination CCX's step lane.

**vnni-product TTFA (§9) — classification.** A profile/backend-selection bug: the lane pins
`QWEN_PREFILL_MATMAT=0` and the no-BF16 policy, routing prefill to f32 BLAS on a CPU with
native `avx512_bf16`. It is not evidence about the steady-state architecture (its STREAM
numbers agree with the provisional profile within 0.02-0.05) and is excluded from every
capacity statement above. Fix = a `vnni-bf16-product` lane (or a backend selection rule:
bf16 prefill wherever `avx512_bf16` is present); owner: profile control plane, not the engine.

## 11. Smallest implementation hypothesis — true decoder/step overlap (design only, no code)

Scope: one prefork worker = one CCX; default-off behind one env (e.g. `QWEN_SD_LANE_SPLIT=4`
= number of decoder cores taken from the worker's mask). Everything else unchanged.

1. **Threads and affinity.** Engine pool `K − split` threads pinned to the first cores of the
   worker's mask (the loop thread among them); decoder private team `split` threads pinned to
   the remaining cores. The private team already exists (`QWEN_SD_POOL=private`,
   `qwen_sd_pool_mode`); what is new is its affinity and that the engine pool is sized
   `K − split`. No oversubscription: 4 + 4 threads on 8 cores.
2. **Handoff.** In the frame loop's decode stage (`qwen_tts.c:3172-3244`), when a slot's
   pending frames reach its target, instead of calling `qwen_speech_decoder_decode_streaming_st`
   inline: copy the chunk's codes into that slot's single-entry mailbox and continue to the
   Talker step. One decoder thread (owner of the private team) services mailboxes; per call
   it runs the unchanged per-item decoder on its team and emits PCM through the existing
   sink path (`on_chunk` → `send_pcm_chunk`, or the OUT writer when enabled).
3. **Order and back-pressure.** Mailboxes are served earliest-playback-deadline first (the
   slot whose delivered audio runs out soonest). If a slot's mailbox is still occupied when its
   next chunk is ready, the loop thread blocks on that slot only until the mailbox frees: the
   lead is bounded to one chunk per slot and no queue can grow. This is the only scheduling
   rule; no work is ever withheld from an idle resource (the difference from the rejected lead
   gate), and no second submitter touches the engine pool (the difference from the rejected
   same-pool consumer).
4. **Finalize path.** The slot's tail decode at EOS goes through the same mailbox; `on_done`
   fires from the decoder thread after the last chunk. Cancel: the loop marks the slot, the
   decoder thread drops the mailbox.
5. **Telemetry.** `[STAGE]` gains `decode_wait_ms` (time the loop blocked on a full mailbox)
   and `[DECODE] placement=LANE`; the cost map's `decoder.total` moves to the decoder thread
   role, which the report already separates.
6. **Falsifier** (the only benchmark to run): `1x8@0-7`, 1.7B, fixed short text, C = 3, 4, 5,
   split 4, versus §8. GO if B4 STREAM p95 ≤ 0.90 with iteration wall p95 ≤ ~110 ms (today
   205) and prebuffer p95 ≤ 250 ms; NO-GO if B3 does not beat 0.90 or TTFA rises. Then the
   same on 4x8 at C12/C16 with the provisional profile, then a SOAK. Nothing else.

Code touchpoints from the audit: `qwen_tts.c` decode stage and FINALIZE/CANCEL macros;
`qwen_tts_speech_decoder.c` `sd_pool_run`/private team creation (add affinity); prefork child
setup in `qwen_tts_server.c` (`sched_setaffinity` slice, `qwen_set_threads`). Estimated size
200-300 lines, all behind the flag.

What changed: §10-11; no PLAN change (the task for the pipeline falsifier is the user's call).
