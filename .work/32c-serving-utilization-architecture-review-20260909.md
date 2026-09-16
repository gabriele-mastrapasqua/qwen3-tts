# 32-core Turin — serving utilization architecture review (2026-09-09)

Status · READ-ONLY REVIEW. No runtime code, PLAN.md, profile or commit was changed.
Evidence added by this review: eight single-worker cost-map censuses on the c8a.8xlarge
(1x8@0-7 at C=1..4 for 1.7B and 0.6B, 1x4@0-3 at C=1/2, 1x16@0-15 at C=4, two chunk-4 arms;
each ~15 s; untracked under `~/bench/ccx/` on the host), the per-worker fields of every wave
JSON of 2026-09-08, and the executed-path census of the 1.7B B2 run. Everything else is the
repository, PLAN.md and the addenda named inline.

Labels: **MEASURED** (a number read from an artifact), **INFERRED** (arithmetic on measured
numbers, stated), **SPECULATIVE** (a model without a measurement behind it). Confidence
HIGH / MEDIUM / LOW is given where a judgment is made.

---

## ONE-PAGE EXECUTIVE SUMMARY

**CURRENT FACT (MEASURED).** On one 8-core CCX of the c8a.8xlarge, one 1.7B worker costs
`T_frame(B) ≈ 44 ms + 10.5 ms × B`: Talker 26 ms and Code Predictor 18-21 ms are a fixed
weight stream at the CCX roof (55 GB/s) and are already batched (B2 costs +1 ms), the
speech decoder is 8.7-9.0 ms per slot per frame and is NOT batched (per item on VNNI). Measured
STREAM_RTF p95 per lane: B1 0.675 · B2 0.832 · B3 0.966 · B4 1.162. Four lanes → C8 (B2) holds,
C12 (B3) is over the 0.90 gate, C16 (B4) is over realtime. 0.6B: `27 ms + 8 ms × B` → B4 0.849,
which is why it reaches C12/C16. Wide pools (1x32, 2x16 beyond B4) add 12+ ms/frame of
rendezvous/idle and collapse; context switches go 4k → 20k → 40k → 90k/s from 1x8 to 1x32.

**CURRENT BOTTLENECK HYPOTHESIS (MEASURED + INFERRED, HIGH).** The machine is
*engine-limited by the per-slot speech decoder*, not by bandwidth. The doctor's physics
ceiling (C28-32) assumes a free decoder; the decoder is 80 % of the marginal cost of every
extra stream. The decoder runs 303 GMAC per 96 frames (almost 2× the Talker's 163 GMAC at B2)
at roughly 45 GMAC/s per core, about 5 % of Zen5 VNNI peak: 84 % of its MACs are on VNNI/bf16
kernels, 16 % on f32 BLAS, and its attention, RoPE, layer-norms, dwconv, residual adds, VQ and
final conv are scalar on the loop thread, with 50-120 pool dispatches per decoder call per
item. It is glue-bound, exactly as P4 found on AMX.

**BIGGEST UNCERTAINTY.** Whether the decoder can be taken off the lane's critical path
(4 cores step-lane + 4 cores decoder inside one CCX) without the two halves stealing each
other's bandwidth. The arithmetic fits (Talker+CP at 4 threads cost the same as at 8:
26.2 + 18.4 ms, MEASURED); the contention term is UNKNOWN and is a 20-second experiment.

**32-CORE UTILIZATION VERDICT.** Mixed: cache-domain-limited for the shape (one CCX = one
lane, 4x8 is right), engine-limited inside the lane (decoder per slot). The host is not
"leaving half the machine on the table" through one serialization point; it is spending a
third to a half of each lane on a decoder that does its work at 5 % of peak and cannot
overlap anything.

**C12 PLAUSIBILITY (INFERRED, MEDIUM).** Yes, with one of: decoder per-slot cost −20 %
(8.7 → 7 ms, B3 lane at ≤ 0.90), or a partitioned decoder (B3 lane at ~0.65 on paper). C12 at
the *hard* gate (< 1.0) is already measured at 0.966 on a short bank; it is not GOOD.

**C16 PLAUSIBILITY (INFERRED, LOW-MEDIUM).** Only with the decoder off the critical path or
halved (B4 lane needs −21 ms/frame to reach 0.90). Not reachable by scheduling.

**BEST NEXT MEASUREMENT.** Two independent 4-core servers on the two halves of one CCX
(1x4@0-3 and 1x4@4-7, 1.7B, C=2 each, 20 s): if each stays at its solo 0.97, the CCX
bandwidth is not contended by two lanes and the partition arithmetic stands; if they degrade,
the partition is dead before a line of code.

**BEST NEXT FALSIFIER (after that).** A default-off partitioned lane: engine pool on 4 cores,
decoder private team pinned on the other 4, asynchronous chunk handoff with a bounded
one-chunk queue; screen 1x8@0-7 at C=3/4 against the census above. Distinct from both
rejected designs (same-pool second submitter; oversubscribed private team).

**DO-NOT-DO LIST.** No lead/credit gate re-run; no same-pool decoder consumer; no cap-3 or
utilization-admission sweeps; no 1x32 or 8x4 for 1.7B; no global cross-worker batching;
no VNNI prepack; no q32; no "implement CP batching" (it exists and works); no long C10+
campaigns until a lane holds B3 at ≤ 0.90; no capacity claim from the physics ceiling.

---

## 0. Reconstructed current state

### 0.1 Sources read

PLAN.md, ENGINEERING.md, 40 `.work` addenda (full list in the agent trace; the ones cited
below by name), `docs/runtime-map-c8a-c4.md`, `docs/reference-aws-c8a-16c-vnni.md`,
`docs/reference-aws-c8i-8c-amx.md`; code: `qwen_tts.c` (serve loop), `qwen_tts_server.c`,
`qwen_tts_talker.c`, `qwen_tts_code_predictor.c`, `qwen_tts_speech_decoder.c`,
`qwen_tts_sd_gemm.c`, `qwen_tts_thread.c`, `qwen_tts_kernels.c`, `qwen_tts_dispatch.c`,
`qwen_tts_costmap.c`, `tools/doctor.py`, `tests/serve_parallel_wave.py`.

### 0.2 Host identity (MEASURED, `profiles/doctor/2026-09-08_220014_*` on the host)

| item | value |
|---|---|
| instance | AWS `c8a.8xlarge` spot, us-east-1b; AMD EPYC 9R45 (Zen5) |
| cores | 32 physical, SMT off (`Thread(s) per core: 1`), 1 socket, 1 NUMA node |
| cache | L2 1 MiB/core; L3 32 MiB × 4 CCX (cpus 0-7, 8-15, 16-23, 24-31) |
| RAM | 61.6 GiB, no swap |
| ISA / build | avx512f/bw/vl/dq, VNNI, BF16; no AMX. `make blas` → `SIMD=avx512bf16`, isa_class `x86_avx512bf16`, gcc 15.2, distro OpenBLAS, source `6497808:clean` |
| gates | SMT off PASS · governor PASS · cgroup PASS · self-test PASS · dispatch gate PASS · cpu-check 18/0 |
| membw read | 231 GB/s @32T (sweep 1T 18 · 8T 141 · 16T 222 · 32T 231) |
| GEMV int8 roof (`roof_matvec_int8`) | mask 0-3 @4T **55.2** GB/s DRAM / 70.4 cache-resident · 0-7 @8T **55.0** / 65.7 · 0-15 @16T **109.9** / **40.5** · 0-31 @32T **208.3** / 333.8 |

The 55 GB/s CCX roof is reached at 4 threads; 8 threads on the same CCX add nothing. The
cache-resident probe on a two-CCX mask (40.5 GB/s) is below its own DRAM rate: a set split
over two L3s thrashes between them. These two facts drive most of what follows.

### 0.3 Evidence table (condensed; every row MEASURED from the named addendum/artifact)

Hosts: **H12** = GCP c4-standard-24, 12 AMX cores, 2x6 · **H8** = GCP c4-highcpu-16, 8 AMX
cores · **T32** = this host · **T16** = c8a.4xlarge 16 Zen5 cores (old stack, docs only).
Columns: STREAM = p50/p95 · TOTAL = p95 · pb = required prebuffer p95 ms · sps = safe-play-start
p95 ms · st = stall@250/@500.

| mechanism / experiment | host · model · topo | C · eff. B | TTFA p95 | STREAM | TOTAL | pb · sps | max gap | st | cores / bw | result · lesson |
|---|---|---|---|---|---|---|---|---|---|---|
| CT-5 clean envelope (q8, cap2) | H12 1.7B 2x6 | C2/C3/C4 · — | 366/547/544 | .600/.623 · .664/.786 · .793/.856 | .638/.867/1.066 | 47·398 / 370·814 / 596·978 | 421/532/586 | 0 · 17%/0 · 75%/25% | — | GOOD/GOOD/MARGINAL (PLAN trusted state) |
| CT-4 Talker B1/B2 | H12 1.7B 1x6 | — | — | — | — | — | — | — | — | 21.6 ms/frame B1; 23.7 ms/iteration B2 → **B2/B1 = 1.10** |
| Quantum floor q1/q2/q4/q8 | H12 1.7B 2x6 C4 | 1.78-1.81 | 424-428 | 1.005/.892/.862/.817 (p95) | — | pb 224/149/277/333 | — | 0/0 | 6.4-6.7 | prebuffer follows quantum; q1 rejected; q4 compromise |
| Fused residual A/B | H12 1.7B q8 C4 | 2.35→2.33 | 179→170 | .831→**.788** | .905→.862 | 389→266 · 568→426 | 521→510 | 17%→0 | 9.0→9.1 | real material-work reduction |
| F1 fused + q4 | H12 1.7B C4 | 2.37 | 175.5 | .827/**.868** | .942 | 201·362 | 344 | 0/0 | 8.8 | next C4 reference candidate |
| F-cap3 | H12 1.7B q4 C5 | 1.57/0.85 | 612.7 | .831/**.969** | 1.257 | 366·980 | 563 | 13.3%/0 | 8.04 | FAIL: fifth proxy 1.028; decoder B3 117-125 ms/call |
| LS-4 utilization admission 40/60/80 ms | H12 1.7B C4+1 | B3 seen | 778-784 | .993/1.004/.985 (p95) | — | 529-801 | 653-704 | 50%/0-17% | UNKNOWN | FALSIFIED predicate; no more thresholds |
| Lead gate | H12 1.7B q8 C3/C4 | 1.48/1.90 | 436/495 | .769→**.986** · .838→**.986** | | | 691/732 | 33%/25% | 6.2→4.1 · 7.3→5.2 | REJECTED implementation: parks 95.8 % of checks |
| Same-pool decoder consumer | H12 1.7B q8 C3/C4 | — | 167→**1200** · 174→**1126** | .833→1.168 · .847→**1.296** | | 308→1534 | 521→1329 | | csw ×2.9 | REJECTED implementation (second submitter, gang lost) |
| Private 2nd decoder team | T16 1.7B 2x8 C4 | — | — | +20-50 % STREAM | | | | | 21 thr / 8 cores | superseded: oversubscribed |
| Prefill helper | H12 C3/C4 | — | 435→**2379** · 503→**2459** | | | sps →2659/2749 | | 33→17 % · 25→50 % | | REJECTED implementation |
| F3 cohort coincidence | H12 q4 cap2 C4 | — | — | — | — | — | — | — | — | useful B≥3 events 2.7 % ±1 ms, 11.7 % ±8 ms → global batching not justified there |
| M-split AMX tasks | H12 chunk32 C4 | — | — | .9707→**1.1056** | | | | | | REJECTED: more tasks = more rendezvous |
| SL-1 known-text | H8 1.7B C1/C3 long | — | 329→65 · 855→158 | | | | 513-538→245-289 | | prefill 244-292→30-37 ms | PROMOTED |
| QL-1 final | H8 1.7B 1x8 cap2 C2 SOAK | — | 155 | p95 .816 | | 258·365 | 345 | 0/0 | | PRODUCTION POINT C2; C3 SOAK windows 1.024/.909/.913/1.006/.874 NOT GOOD |
| QL-2 1x8 cap4 | H8 1.7B | C1-C4 | 71/125/184/236 | .480/.663/.784/**.974** (p95) | .500/.713/.863/1.087 | 0/40/111/242 | | 0/0 | 6.7-6.8 | doctor rho .75 falsified (C4 .974); **≈12 ms per slot** (judge) |
| 0.6B product | H8 0.6B 1x8 cap3 C3 SOAK | — | 185 | p95 .861 | | 377·471 | | 2 %/0 | | GOOD; C4 ~.943, C5 ~1.077 |
| T16 old stack 2x8 C4 soak | T16 1.7B | — | 193-205 | .993-1.006/1.057-1.069 | | | | | admission 108-110 ms ≈10 % wall; pool asleep ≈25 % | pre-SL-1, pre-fused |
| Turin screen 4x8 cap2 | T32 1.7B | C4/6/8 · 1.68/2.66/3.35 (in-flight) | 106/173/181 | .71/.87/.87 (p95) | .75/.95/.95 | 67/327/334 | 426-569 | 0 · 0 · 12 %/0 | 20.9-23.1 | C8 marginal |
| Turin warm ×3 | T32 1.7B 4x8 C8 | 3.46/3.32/3.38 | 187/174/172 | .92/.87/.90 | 1.00/.94/.97 | 357/295/344 | | 25/0/12 %@250 | 21.9/21.3/21.6 | warm = −10-15 ms TTFA only |
| Turin chunk4 + decoder-batch 1 | T32 1.7B 4x8 C8 | 3.44/3.37 | 180/176 | .92/.89 | .99/.97 | **228/188** | 301/275 | 0/0 @250 | 21.9/21.4 | prebuffer −100 ms, RTF unchanged |
| Turin 2x16 cap4 | T32 1.7B | C4 · C8,C8 | 115 · 223/210 | .53 · **.84/.79** (p95) | .58 · .94/.91 | 29 · 276/189 | 372-475 | 0 · 0/0 @250 | 24.0/23.0/23.8; csw 37-40k/s | best C8 |
| Turin 2x16 cap5 c4db1 | T32 1.7B | C8 · C10 | 224 · 270 | .90 · **1.05** | 1.00 · 1.17 | 237 · 346 | | 0/0 | 22.4/22.5 | C10 over realtime |
| Turin 1x32 cap8 | T32 1.7B | C4 · C8,C8 | 246 · 418/386 | .67 · **1.55/1.40** | .79 · 1.73/1.56 | 124 · 1287/1084 | 809/714 | 0 · 62 %@500 | 22-23; csw 75-91k/s | dead |
| Turin 0.6B 4x8 cap4 | T32 0.6B | C8/12/16 · 3.9/6.0/8.4 | 103/133/168 | .63/**.72**/.83 | .64/.74/.86 | 73/155/373 | 440-613 | 0 · 0 · 25 %@250 | 23.6/25.4/26.2 | C12 @250, C16 @500 |
| Turin 0.6B 2x16 cap8 | T32 0.6B | C8/12/16 · 4.1/6.8/10.0 | 133/186/253 | .68/.88/**1.22** | .69/.93/1.28 | 132/485/1129 | | 0 · 58 % · 100 % @250 | 22.9-24.4; csw 48-55k/s | wide pool collapses |
| **Single-CCX census 1x8@0-7 (this review)** | T32 1.7B | C1/2/3/4 | 93/175/254/337 | .675/.832/.966/**1.162** (p95) | .694/.875/1.089/1.302 | 42/298/531/809 · 135/473/767/1146 | 410-651 | — | 7.2-7.5 (of 8); csw 2.0-4.2k/s | the lane law, §4 |
| Single-CCX census 1x8@0-7 | T32 0.6B | C1/2/3/4 | 55/96/136/172 | .437/.593/.703/.849 | .445/.607/.722/.872 | 0/44/162/428 · 55/140/298/600 | 265-532 | — | 7.0-7.4 | |
| 1x4@0-3 | T32 1.7B · 0.6B | C1 · C2 · 0.6B C1 | 107 · 215 · 61 | .726 · .972 · .481 | .749 · 1.027 · .491 | 105 · 531 · 7 | | | 3.7-3.8 | Talker+CP identical to 8T; decoder 12.9 ms/frame |
| 1x16@0-15 | T32 1.7B | C4 | 214 | .760/.797 | .890 | 248·462 | | | 13.0; csw 20.6k/s | ~12 ms/frame unattributed, §5 |
| 1x8@0-7 chunk4 + db1 | T32 1.7B | C2 · C3 | 178 · 254 | .850 · **1.044** | .902 · 1.161 | 179 · 312 | | | | chunk 4 = +2-3 ms decoder/step, −120 ms prebuffer; decoder-batch=1 is a no-op kernel-wise on VNNI |

Concept vs implementation, as the addenda themselves state it: lead gate, same-pool
consumer, prefill helper, one-row direct quant, BLAS-C residual, scratch reuse, VNNI prepack →
**one implementation rejected**. q32, low-N M split, cap3 on H12, LS-4 predicate → **rejected
for that host/workload**. Global cross-worker batching → **not justified on H12** (2x6),
explicitly host-conditional. Partitioned step/decoder team → **never tested**, gated on
"arithmetic fits + second host reproduces the ENGINE/MIXED signature".

### 0.4 What the wave's "B" column is (MEASURED from code)

`batch_eff` is the prefork parent's `mean_inflight`: the time-weighted number of requests in
flight box-wide between dispatch and socket close, sampled over the whole level window
including the 1.2 s pauses and tails (`qwen_tts_server.c:2907-2926`). "B 3.35 at C8 on four
workers" therefore means 0.84 requests per worker averaged over the window, not the per-step
batch. The engine's per-iteration batch (`[shape-census] frames=(single/batched/…)`,
`[serve-profile] decode occupancy`) is the right counter; in the single-CCX C=2 census it
shows 15 of 30 measured steps at B=2 and 15 at B=1: the two texts of the short bank differ in
length, so the pair decays to B=1 for half its life. **Effective batch in these waves is
limited by text-length dispersion, not by batch formation** (INFERRED, HIGH).

---

## 1. The 32-core fact pattern, verified

| claim in the brief | verified value | source |
|---|---|---|
| ~231 GB/s host bandwidth | 231 read @32T (membw); 208 GB/s int8 GEMV roof @32T | doctor 22:00 |
| ~55-56 GB/s per 8-core domain | 55.0 (0-7), 55.2 already at 4 threads (0-3) | doctor |
| 4x8 warm C8 STREAM p95 ~0.87-0.90, TOTAL ~0.94-0.97, TTFA ~170, pb ~300 | 0.87/0.90/0.92 · 0.94/0.97/1.00 · 172-187 · 295-357 | warm-4x8 |
| C10 STREAM p95 ~1.05, TOTAL ~1.17 | 1.049 / 1.172 (2x16 cap5 c4db1) | beyond-c8 |
| 0.6B 4x8 C4/6/8 STREAM ~.45/.60/.59 | .445/.597/.587 p95 | step1-06b |
| 0.6B C12 ~.72, C16 ~.83 | .715 / .829 p95; C16 stall@250 25 %, @500 0 % | 0.6b-4x8-cap4 |
| 0.6B 2x16 C16 much worse | 1.224 p95, prebuffer 1129 ms | 0.6b-2x16-cap8 |

All verified. Caveat that applies to every Turin number: provisional profile
(`aws-c8a-16c-vnni-ttfa`: q8, `QWEN_DECODER_BATCH=0`), one wave, short bank only, no
`--language`, batch caps set by hand. These are screens.

The 0.6B proves what the brief says it proves: the server architecture has no universal B2
ceiling; a 4x8 worker runs B3/B4 with no scheduler pathology (0.6B B4 at 0.849 per lane).
What differs between the models is only the fixed weight stream (§4).

---

## 2. Physics ceiling vs engine ceiling — decomposing the gap

Doctor section 8 on this host (MEASURED roofs, PREDICTED ceilings):

| model · shape | roof GB/s | physics C (B) | model C (B) | floor | measured cliff |
|---|---|---|---|---|---|
| 1.7B 1x32 | 208 | 16 (16) | 16 (16) | 1 | C8 at 1.40 (FALSIFIED) |
| 1.7B 2x16 | 110 | 32 (16) | 12 (6) | 2 | C8 0.84 · C10 1.05 |
| 1.7B 4x8 | 55 | 28 (7) | 4 (1) | 4 | C8 0.87 · lane B3 0.966 |
| 0.6B 2x16 | 110 | 32 (16) | 16 (8) | 2 | C12 0.88 · C16 1.22 |
| 0.6B 4x8 | 55 | 64 (16) | 12 (3) | 4 | C12 0.72 · C16 0.83 |

The physics number is "Talker + CP bytes at the GEMV roof, B2/B1 = 1.10 per extra slot,
decoder free". On one CCX that is 44 ms fixed + ~1-2 ms per slot → B7 in 80 ms. The
measured lane (§4) is 44 ms fixed + **10.5 ms per slot**. So the entire gap between C28 and
C8 on 4x8 is the per-slot term, and the per-slot term is:

| component of the ~10.5 ms marginal slot cost (1.7B, 8T) | ms | label |
|---|---|---|
| speech decoder, per item, inline | **8.7-9.0** | MEASURED (cost map `decoder.total` / frames) |
| Talker per-slot work inside the region (norms, RoPE, KV store, attention, SwiGLU, quant) + B-scaling of the VNNI matmat | ~1.0 | MEASURED (26.2 → 27.3 → 27.6 → 29.0 ms at B1..B4, diluted by B-mix) |
| CP per-slot work inside the region (same list ×16 steps + 15 argmaxes) | ~1-1.5 | MEASURED (18.1 → 21.0 → 21.4 ms; step from B1 to B2, then flat) |
| loop-thread serial: logit clamp + softmax/top-k/top-p, embedding accum (15 bf16 rows), PCM conversion + 3 `write(2)` per chunk, cancel poll | ≤ 0.5-1 | INFERRED (not in the cost map; residual of STREAM vs sum of regions, §4) |
| admission prefill (58 ms per request, inline, stalls the lane) | ~2 ms/frame at B4 over a 3.4 s request | MEASURED per request, INFERRED per frame |

Ranking of the candidate components from the brief, by evidence on this host:

1. **Decoder per-item work** — MEASURED, dominant (80 % of marginal cost). HIGH.
2. **Wide-team efficiency collapse** (rendezvous/idle on 16-32 thread teams) — MEASURED as
   ~12 ms/frame unattributed at 16T and csw 20k-90k/s; kills 1x32 and 2x16 past B4. HIGH.
3. **Talker weight traffic** — MEASURED 26 ms/frame at the CCX roof; fixed, already batched.
   It is the floor, not the gap. HIGH.
4. **CP weight traffic + sequential 16-step structure** — MEASURED 18-21 ms; fixed, batched;
   the 16 steps are 718 spin barriers per frame, all inside one pool entry. HIGH.
5. **Cross-CCX cache traffic** — MEASURED at the probe level (40.5 GB/s cache-resident on 16T)
   and at the topology level; not separable from (2) in the serving numbers. MEDIUM.
6. **Admission/prefill interference** — MEASURED 58 ms per request inline; ~2-7 % of the lane
   at C3-C4 with short texts, larger with long ones (SL-1 pins it at ~73 ms flat). MEDIUM.
7. **Per-slot serial tax outside regions** (sampling, embed, output) — INFERRED ≤ 1 ms/slot.
   LOW as a lever.
8. **CP effective batching / batch formation** — NOT a cause: batched at B_eff every frame,
   B_eff = n_active (§3). Dispersion of text lengths lowers the time-averaged B, which lowers
   *throughput accounting*, not the lane's frame time. HIGH.
9. **Talker compute**, **thread oversubscription** (none: K-1 workers + caller per process),
   **socket/output** (3 writes per chunk), **LLC capacity for CP** (112 MB never fits a 32 MB
   L3; CP runs at DRAM rate on every shape) — not the gap.

Theoretical FLOPS were not used anywhere above; the decoder's 5 %-of-peak figure is a
ratio of two measured numbers (GMAC from the census, ms from the cost map) and is offered as
a characterization, not as a target.

---

## 3. Code Predictor claim — audited

Mandatory answers, from `qwen_tts_code_predictor.c` and the run logs.

1. **Is CP batched across active streams?** Yes. With B_eff ≥ 2 the whole 16-step frame runs
   as one persistent parallel region, `cp_frame_region_run` (`cp.c:1377-1427`), logged as
   `[cp] whole decode frame as one parallel region: ON (team K, BW n)`.
2. **Logical B:** `B_eff` = the number of active slots this iteration (`qwen_batch_pack_active`,
   `talker.c:2045-2056`); no narrowing in the profile. "BW 4" in the log is the B_eff of the
   first batched frame, printed once.
3. **Kernel B:** the same B_eff, 2..16, passed to `qwen_region_i8_run` (VNNI row-block
   matmat; tiles m4n4 for B ≤ 4, m2n4 for 5..8, plain rows above 8). One weight traversal for
   all slots per projection.
4. **Batched:** MTP projection, per layer QKV (fused), O, gate/up (fused), down, and the 15
   codebook lm_heads (vocab 2048 × ch).
5. **Per slot, one slot per thread between barriers:** embedding lookups (`cp_codec_emb`),
   bf16→f32, RMS norms, RoPE, KV store into `bb->cp_kv_k/v[slot]`, single-query attention over
   ≤ 64 positions, SwiGLU, argmax. With B_eff = 2..4 on a team of 8-16, the remaining threads
   spin at the barrier during these sections.
6. **One traversal per step?** Yes: each of the 16 steps performs one batched weight traversal
   of the 5 layers for all B slots. The 112 MB working set is read 16× per frame *per worker*,
   not per stream. The cost map confirms it: `cp.decode.total` is 18.1 ms/step at B1 and
   21.0-21.4 ms/step at B2..B4 (MEASURED).
7. **What `region.cp`, `region.cp_frame`, `region.cp_batch_head` mean:** `cp` = the
   transformer step as one region (per step); `cp_frame` = all 16 steps in one pool entry
   (the path that runs here); `cp_batch_head` = MTP projection and lm_heads done once for
   all slots as dispatched matmats on the non-frame-region fallback path (`cp.c:1220-1268`).
   The dispatch-map's "see reason" is a runtime-resolved ON here.
8. **Effective CP B in the C8 4x8 runs:** per iteration, equal to the number of active slots
   in the worker (2 while both requests live). Time-averaged over the window: 0.6-1.1 per
   worker (wave JSON). Per step during overlap: 2 (INFERRED from the lockstep code path).
9. **Why the reported B is 3.35:** it is the parent's window-averaged in-flight count (§0.4).
10. **Batch formation limited by phase/cadence?** No: lockstep, every active slot steps every
    iteration; the batch is re-formed each iteration from the active set. Limited only by
    request lifetime dispersion.
11. **Does decoder completion desynchronize streams and destroy the next CP batch?** No. The
    decoder call is inline on the loop thread *before* the Talker step of the same iteration;
    all slots wait for it. The batch is intact; the cost is that the lane idles its other slots
    for 8.7 ms per decoded slot.
12. **Sampling/state/head at B3/B4:** the codec head is one bf16 matmat (weights read once);
    sampling is per slot on the loop thread (softmax over vocab, top-k/top-p). Not measured
    separately; bounded by the residual in §4 (≤ ~1 ms/slot).
13. **Is CP bandwidth-bound at B2..B4?** Yes: 1.8 GB per frame per worker at 55 GB/s = 33 ms
    if fully DRAM; measured 18-21 ms means part of the set is served from L3/L2 across the
    16 steps (the 5-layer body re-read 16× keeps some rows resident). It does not fit the
    32 MB L3 (question 14: **no**), and it stays flat with B.
15. **Does widening to 16/32 threads hurt it?** The batched CP on 16T measured ~7.6 ms/step
    average (1x16@0-15, MEASURED, diluted), i.e. it *speeds up* with two CCX of bandwidth; what
    widening hurts is the rendezvous around it and the cache-resident phases (40.5 GB/s).

**Actual execution diagram** (one worker, one iteration, B_eff active slots; MEASURED
ms at 1.7B on 8 threads, B2 mix, from the cost map; barrier counts from code):

```
iteration k  (loop thread "srv-sched"; engine pool = K-1 workers + caller)
├─ admission scan: prefill of a new request INLINE (58 ms, stalls every slot)  [per request]
├─ codec head        bf16 matmat over B_eff, 1 dispatch                       [batched]  ~1 ms
├─ sampling          per slot on the loop thread                                [serial]   <1 ms
├─ CP frame region   1 pool entry, 16 steps × (MTP + 5 layers + lm_head)        [batched]  18 → 21 ms
│                    718 spin barriers; projections = 1 traversal for all B
├─ embed + decoder   per slot: 15-row embedding accum; every q frames ONE       [per item] 8.7-9.0 ms × B
│                    per-item decoder call (int8 VNNI residual convs, f32 BLAS   (per decoded
│                    ConvT/transformer slices, scalar attention/norm/dwconv),    slot-frame,
│                    50-120 pool dispatches per call; PCM write inline           amortized)
├─ Talker region     1 pool entry, 28 layers, 225 spin barriers                 [batched]  26 → 27-29 ms
└─ pos++             (next head uses this hidden)
```

Batching is real for the head, the CP and the Talker. It collapses, by construction, at the
decoder (per item on VNNI whatever `QWEN_DECODER_BATCH` says: `sd.c:3852-3856` routes to
`sd_batch_fallback`, a sequential per-item loop) and at everything on the loop thread.

---

## 4. 0.6B as the Amdahl control — MEASURED on one CCX

Single worker, `1x8@0-7`, cap 4, C = B, short bank, one wave, profile q8 / decoder-batch 0,
cost map level 1. Per-step costs are derived from `serve`-role inclusive regions after
subtracting the two warm-up requests (27 frames each at B1 cost), divided by the measured
number of batched steps (30 at C2/C3, 42 at C4). Because texts differ in length, the "B"
levels are mixtures (C2 ≈ 50 % of steps at B2); true fixed-B costs are slightly higher than
the diluted figures at B3/B4.

**1.7B, 8 threads**

| | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Talker step (ms/step) | 26.2 | 27.3 | 27.6 | 29.0 |
| CP frame (ms/step) | 18.1 | 21.0 | 21.4 | 21.4 |
| CP head/serial (lm_head inclusive) | 0.6 | 0.4 | 0.4 | 0.4 |
| Decoder (ms/step, all slots) | 9.0 | 17.4 | 23.7 | 25.8 (diluted; ~8/slot) |
| Pool wait / barrier | UNRESOLVED (spin barriers inside regions are inclusive in the region time; `pool_dispatch` counters 0 at level 1) | | | |
| Other serial (residual: measured frame − sum) | ~1 | ~1 | ~4 | ~13 (incl. 4 inline prefills ≈ 5.5 ms/frame) |
| **Sum of regions** | **53.3** | **65.7** | **72.7** | **~81** (undiluted ≈ 44 + 4×10.5 = 86) |
| **Measured STREAM p95 × 80 ms** | **54.0** | **66.6** | **77.3** | **93.0** |
| effective B (per step, during overlap) | 1 | 2 | 3 | 4 |
| GB/s (Talker weights 1.42 GB / step) | 54 | 52 | 51 | 49 |
| core-equivalents (window mean, of 8) | 7.5 | 7.3 | 7.2 | 7.2 |
| deadline margin vs 80 ms (measured) | +26 | +13 | **+3** | **−13** |
| margin vs the 72 ms preferred gate | +18 | +5 | **−5** | **−21** |

**0.6B, 8 threads**

| | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| Talker step | 8.8 | 10.2 | 9.6 | 10.8 |
| CP frame | 17.3 | 18.9 | 20.5 | 21.5 |
| Decoder (all slots) | 8.8 | 14.2 | 22.0 | 26.8 (diluted) |
| Sum of regions | 34.9 | 43.3 | 52.1 | 59.1 |
| Measured STREAM p95 × 80 | 35.0 | 47.4 | 56.2 | 67.9 |
| deadline margin (measured) | +45 | +33 | +24 | +12 |

**4 threads (mask 0-3), 1.7B B1:** Talker 26.2, CP 18.4, decoder **12.9** ms/frame; STREAM p95
0.726. **1.7B B2 on 4T:** Talker 27.8, CP 23.2, decoder 24.3 (≈ 12/slot); STREAM p95 0.972.
**0.6B B1 on 4T:** Talker 8.6, CP 17.5, decoder 12.5.

**Reading (INFERRED, HIGH).**
- The Talker shrink from 1.7B to 0.6B removes 17.4 ms of *fixed* cost (26.2 → 8.8), which is
  0.97 GB fewer weights per frame at 55 GB/s = 17.6 ms. The weight stream is the fixed floor
  and it is entirely bandwidth.
- The marginal cost per slot is **10.5 ms (1.7B) vs 8 ms (0.6B)**, of which the decoder is
  8.7 vs 7 ms. The decoder is the same model in both; its per-slot cost is the
  weakly-scaling floor after the Talker shrinks. On 0.6B at B4 the decoder is 45 % of the
  frame, the CP 36 %, the Talker 18 %.
- What prevents a 1.7B lane from reaching B3 at the preferred gate is 5 ms; B4 is 21 ms. The
  stage holding those milliseconds is the decoder: 3 × 8.7 = 26 ms at B3, 35 ms at B4, all
  serial on the lane, all while the other slots' Talker/CP cannot proceed.
- The 4-thread lane confirms the bandwidth argument: Talker+CP cost the same on 4 cores as on
  8 (26.2 + 18.4 vs 26.2 + 18.1). Only the decoder pays for fewer cores (8.7 → 12.9, ×1.48 for
  half the threads: it scales sub-linearly, as CT-2 found on AMX).

**Instrumentation needed for a cleaner table (none of it new code):** `QWEN_STAGE_TRACE=1`
gives per-iteration `active step head sample cp decode talker output serial` ms, which
removes the B-mix dilution; `QWEN_SD_PHASE=1` splits the decoder call; `QWEN_COST_MAP=2` adds
the level-2 CP sub-regions; `-DQWEN_POOL_STATS` exposes park/dispatch counts. A fixed-length
bank (one text repeated) would make every step a true B-level.

---

## 5. Why 4x8 works and 2x16 / 1x32 collapse

MEASURED facts:
- GEMV roof: 55 GB/s per CCX at 4 threads; 110 at 16 threads across two CCX; 208 at 32. A
  cache-resident 101 MB set streams at 66-70 GB/s on one CCX, **40.5 GB/s** across two, 334
  across four (each CCX holding a quarter). Splitting a set that does not fit one L3 over two
  L3s is slower than DRAM.
- Context switches per second (wave JSON, whole box): 4x8 8-9k · 1x8 2-4k · 1x16 20.6k ·
  2x16 37-42k · 2x16 (0.6B cap8) 48-55k · 1x32 75-91k. Spin budget is 4096 `pause`
  iterations before parking (`thread.c:380-401`); a 16-32 thread team parks and wakes far
  more often per frame than an 8-thread one.
- 1x16@0-15 at B4: Talker 18.9 ms/step, CP ~7.6 ms/step (MEASURED, diluted): the fixed
  weight stream *does* halve with two CCX. Decoder per slot ~5.5 ms (vs 8.7 at 8T). Sum ≈
  52 ms; measured frame 64 ms (STREAM p95 0.797): **~12 ms/frame unattributed** = rendezvous,
  park/wake and per-slot sections where 12-14 of 16 threads spin (INFERRED, MEDIUM).
- 2x16 cap4 C8 (B4 per worker): STREAM p95 0.79-0.84 — better than 4x8 at B4 (1.16 on the
  lane) because the fixed stream is halved. At B5 (C10) it is 1.05: the marginal slot on a
  16-thread worker measured ~+17 ms (INFERRED from 0.84 → 1.05), steeper than on 8T (10.5),
  which is consistent with a per-item decoder whose dispatch cost grows with team width
  and with the 5-slot per-slot sections leaving 11 threads idle.
- 1x32: 32-thread team, four CCX, one loop thread admitting 8 prefills inline, ~925 spin
  barriers per frame across four L3s, csw 75-91k/s. STREAM 1.40-1.55, prebuffer > 1 s.

Explanation (INFERRED, HIGH for the direction, MEDIUM for the split between the two halves):

1. **Bandwidth is per CCX, so 4 independent lanes get 4 × 55 = 220 GB/s of Talker/CP stream,
   the same aggregate as one 32-thread team (208).** Wide teams gain nothing on the weight
   stream that four lanes do not already have.
2. **Wide teams pay for what lanes avoid**: every per-slot section (norms, RoPE, attention,
   SwiGLU, quant, argmax — 8 per layer, 943 barriers per frame) runs on min(B, team) threads
   while the rest spin; with B = 2-8 and team = 16-32, most of the machine spins during those
   sections. A lane of 8 wastes at most 6 threads; a team of 32 wastes 24-30.
3. **Cross-CCX cache traffic** for anything that would be L3-resident on one CCX (CP rows,
   KV, decoder activations): the 40.5 GB/s probe is the physical signature.
4. **Park/wake**: the spin budget is sized for an 8-thread team; the 16-32 thread teams park
   and wake tens of thousands of times per second, each a futex round trip on the critical path.
5. **One loop thread for all slots** on 1x32: admission prefill of 8 requests, 8 samplings, 8
   inline decoder calls (each 50-120 dispatches) all serialized on one thread while 31 others
   wait. The lane model bounds this serial tax to ≤ 4 slots per thread.

Not the reason: frequency (unreadable on this VM), NUMA (one node), weight duplication
(§13), OpenBLAS threading (held at 1, sliced on the engine pool).

---

## 6. The "natural 4x8 lanes" hypothesis — tested

Hypothesis: `C_total = 4 × B_good_per_lane`; move a lane from B2 to B3/B4.

Attempts to falsify it:

| confounder | test | verdict |
|---|---|---|
| current prefork implementation | the single-CCX census used one non-prefork worker pinned by `--cpu-mask`; its B2 (0.832) matches the 4x8 prefork C8 per-worker figure (0.87-0.90 with 4 admissions of interference) | not the cause; lane law holds without prefork |
| model replication / cache | int8 weights built pre-fork, COW-shared (§13); a lane's Talker runs at 49-54 GB/s of its 55 roof | not the cause |
| measurement artifact | STREAM p95 at B1..B4 is reproduced by the sum of cost-map regions within 1-4 ms at B1..B3 | not an artifact |
| admission pattern | synchronized waves are the worst case for inline prefill (all admitted in the first iterations); at B3/B4 they add 4-6 ms/frame; the lane law is 44 + 10.5·B *before* that | shifts the number by ≤ 6 ms, does not change the verdict |
| batch cap | caps were ≥ C in every census | not binding |
| lucky short bank | short texts (7 chunks) *penalize* the lane (start-up and prefill are a larger share); a long bank would lower the per-frame admission share and raise the decoder share; the SL-1 pin (73 ms flat prefill) is not verified in these arms | direction favors the hypothesis; magnitude UNKNOWN on long texts |
| memory placement | one NUMA node; weights first-touched by the parent | moot here |
| **2x16 as the better shape** | at C8 2x16 measured 0.79-0.84 vs 4x8 0.87-0.90 (both B ≤ 4 per worker) because two CCX halve the fixed stream; but at C10 it is 1.05 and the marginal slot is steeper | **partial falsification**: for the *current* engine, 2x16 cap 4 is the best C8 shape on this host; the lane model is the right one for *raising B*, because a 16-thread team's per-slot cost grows with its width |

Verdict (INFERRED, MEDIUM-HIGH): the hypothesis survives as the *optimization* model. The
CCX is the natural lane; the two things that make a lane fail at B3/B4 are inside the lane
(decoder per slot, inline prefill), not between lanes. Two corrections to the brief's framing:
(a) with the engine as it is, 2x16 cap 4 is a better C8 operating point than 4x8 cap 2 and
should be the screen reference on this host; (b) "B_good" must be read against the 0.90
preferred gate: today B2 at 0.83 is GOOD, B3 at 0.966 is not.

---

## 7. The most informative single-CCX experiment — designed and (mostly) run

Already executed in this review (~2.5 minutes total): the eight censuses of §4 plus the 4T
and 16T probes. What remains to make it the canonical Tier-0 experiment (≤ 60 s per model):

```
1x8@0-7, cap 4, C = 1,2,3,4 on one server (levels repeated twice), ONE fixed-length text
  env: QWEN_STAGE_TRACE=1 QWEN_SD_PHASE=1 QWEN_COST_MAP=2 QWEN_SHAPE_CENSUS=1
  read: [STAGE] per-iteration head/sample/cp/decode/talker/output/serial ms and `active`
        [SDPHASE] per decoder call (vq/preconv/inproj/pretf/outproj/conv split)
        cost map cp.batch.* split and runtime.pool_dispatch ticks
        [serve-profile] decode occupancy histogram (true per-step B)
```

Expected table shape is exactly §4's, with three columns filled that are UNRESOLVED today:
pool wait (needs `-DQWEN_POOL_STATS` or `[STAGE] queue_wait_ms`), other serial (`[STAGE]
serial_ms`), and the decoder split (`[SDPHASE]`).

The headline number is already known:

    B3 budget deficit for 1.7B (one 8-core lane) = 5 ms against the 72 ms preferred gate
        (measured 77.3 ms; hard 80 ms gate met by 3 ms)
    B4 budget deficit = 21 ms against 72 ms (measured 93 ms; 13 ms over the hard gate)

and the stage that contains it is the decoder: 3 × 8.7 = 26 ms at B3, 4 × 8.7 = 35 ms at B4,
serial on the lane. Removing 5 ms means decoder −20 % per item, or taking one slot's decode
off the lane's critical path; removing 21 ms means decoder −60 % per item, or overlap.

---

## 8. Partitioned execution — as a falsifiable model

Why the previous attempts failed (from the addenda, MEASURED): `QWEN_DECODER_THREAD=1` was a
second submitter on the *same* engine pool, serialized on `submit_mtx` behind the Talker/CP
regions, with `dec_batch` forced to 0 (lost the AMX gang); consumer calls stretched to ~1.4 s.
The c8a private second team ran 21 threads on 8 cores. Both reject *those implementations*;
the addenda say so explicitly and gate the partitioned team on "the arithmetic fits".

The arithmetic on this host (MEASURED terms):

| term | 8T lane today | 4T step-lane + 4T decoder-lane |
|---|---|---|
| T_step(B) = Talker + CP | 44 + ~2·(B−1) ms | **T_step(4c)**: 44.6 at B1, 51 at B2 (MEASURED 1x4@0-3); ~53-55 at B3/B4 (INFERRED) |
| D(B) decoder, all slots | 8.7·B ms | **12.9·B** ms at 4T (MEASURED B1, B2 24.3) |
| iteration | T_step + D | max(T_step, D) + handoff |
| B2 | 66 (0.83) | max(51, 26) = 51 (0.64) |
| B3 | 77 (0.97) | max(53, 39) = 53 (0.66) |
| B4 | 93 (1.16) | max(54, 52) = 54 (0.68) |
| B5 | ~104 | max(55, 65) = 65 (0.81) |
| B6 | ~114 | max(56, 77) = 77 (0.97) |

On paper a partitioned lane holds B4 at ~0.7 and B5 at ~0.8: C16-C20 on four lanes. The
brief's conditions, checked:

- *T_step(4c) does not explode*: MEASURED, it does not move (bandwidth-bound at 4 threads).
- *decoder(4c) fast enough*: 12.9 ms/slot-frame → B4 = 52 ms < 80. MEASURED term, INFERRED
  sum.
- *bandwidth does not become worse*: **UNKNOWN**. The 4T roof was measured with the other
  four cores idle. The decoder moves f32 activations (im2col, ConvT taps) through the same
  L3 and memory path; the census says it is compute/glue-bound (5 % of peak), which argues
  for low bandwidth pressure, but it is not measured.
- *cache domains stay local*: yes by construction (both halves inside one CCX).
- *pipeline latency does not hurt TTFA*: the first chunk's decode cannot overlap anything
  before it; TTFA unchanged. Steady-state chunks are delivered when their decode ends, no
  later than today. INFERRED, HIGH.
- *established streams gain cadence*: yes, the lane's Talker/CP no longer stall for other
  slots' decodes; prebuffer should fall with the frame-time variance. INFERRED, MEDIUM.
- *no unbounded queue*: a one-chunk bounded handoff per slot; if the decoder lane falls behind,
  the step lane must block (back-pressure), not queue.

Verdict: **mathematically plausible; not rejected by any prior experiment; contention term
unmeasured.** The ONE cheapest falsifier, before any code:

    two independent servers, 1x4@0-3 and 1x4@4-7, 1.7B, C=2 each, same bank, 20 s
    signature if partition is viable : each lane at ~0.97 (its solo value), Talker ≥ 50 GB/s
    signature if not                  : both degrade toward 1.1+, Talker ms/step rises

This over-approximates the partition's contention (two full lanes each stream 1.4 GB of
Talker per frame; the partition streams it once), so a pass is conservative. It is a Tier-1
experiment (< 1 min) and unlocks whether to build the default-off partitioned lane at all.

Not proposed: any rewrite of the pool, any global decoder service, any change to admission.

---

## 9. CPU-native stream ownership models — judged against the workload

Quantities to keep in view (MEASURED): Talker weight stream 1.42 GB/frame per *worker*,
shared by its slots at +1 ms per slot; CP 1.8 GB re-read per worker-frame; decoder 8.7 ms per
slot-frame at 8T, 12.9 at 4T, sub-linear in threads; CCX roof 55 GB/s = 4.4 GB per 80 ms.

| model | bandwidth check | decoder check | verdict |
|---|---|---|---|
| **A. 4 lanes × 8 cores, ≤ 3-4 streams each, local batching only** (= today's 4x8) | 3.2 GB of 4.4 per lane-frame: fits | 8.7·B serial: B2 ok, B3 marginal, B4 no | current; the reference for any change |
| **B. 8 groups × 4 cores** (8x4) | two lanes per CCX = 6.4 GB per 80 ms per CCX > 4.4 available | decoder 12.9/slot | **infeasible for 1.7B** (doctor 8x4 B1 rho 1.02 agrees); feasible for 0.6B only at B1 (rho 0.80 predicted) |
| **C. ~2 dedicated cores per stream + shared batched Talker/CP** | shared step-lane on 4 cores per CCX streams weights once: fits | decoder on 2 cores ≈ 12.9 × (4/2)^0.6 ≈ 20 ms/frame per stream (SPECULATIVE scaling) < 80: fits per stream | = the partition of §8 generalized to per-stream decoder cores; 4 step + 4 × 1-core decoders would need the decoder at ≤ 80 ms/frame on one core (≈ 8.7 × 8^0.6 ≈ 30 ms, SPECULATIVE). Worth one measurement (`1x1@0` 0.6B B1) |
| **D. producer/consumer stages with bounded deadline queues** | as C | as C | the §8 falsifier is the minimal instance |
| **E. hybrid: logical ownership + opportunistic coalescing** | today's lockstep already coalesces everything coalescible (head, CP, Talker) at no wait | — | nothing to add until the decoder is off the critical path |

Trade-off quantified: dedicating cores to streams (B) multiplies the weight stream by the
number of lanes per CCX and breaks at two; dedicating cores to *decoders* (C/D) costs no
extra weight traffic and only pays the decoder's sub-linear thread scaling. The CPU-native
model that fits the numbers is "one step-lane per CCX, decoders on the CCX's spare cores".

---

## 10. Deadline scheduling view

Per active stream the deadline is the playback lead; the engine today steps every slot every
iteration (lockstep), so the only scheduling decisions are admission (inline prefill) and
decoder cadence (q). Within a lockstep lane with a fixed weight stream, reordering work does
not reduce work: EDF over slots that all need the same Talker step is a no-op, and *parking*
a slot to favour another was the lead gate's mechanism, which lost 95.8 % of its checks and
dropped useful CPU (MEASURED, H12).

Where a deadline view becomes material (INFERRED, MEDIUM): only once the decoder is a
separate lane. Then the decoder lane has real choices — which slot's chunk to decode first
(earliest playback deadline), and whether to accept a one-chunk lead — and the step lane never
waits. That differs from the rejected gate in kind: it orders *ready* work on a second
resource instead of withholding work from the only resource. Until then, deadline scheduling
is reshuffling.

---

## 11. The per-slot serial tax

From the single-CCX census (MEASURED regions, INFERRED residual):

    T_serial(B) ≈ 1 ms + B × (≤ 0.5-1 ms)      outside the regions
    T_slot_tax(B) ≈ B × 10.5 ms                 including the decoder

Itemized (code audit): sampling (softmax/top-k/top-p over vocab 3072-ish per slot, loop
thread), 15-row bf16 embedding accumulation, EOS masking, cancel `poll()` per slot per frame
(if enabled), PCM float→int16 + 3 `write(2)` per chunk, `talker_norm` for every slot after the
region, KV pointer bookkeeping. Inside the regions: per-slot norms/RoPE/attention/SwiGLU/
quant/argmax executed one slot per thread between 943 spin barriers per frame — their cost
is the ~1 ms/slot growth of the Talker and CP regions.

The tax that "erases batching gains" is not this list; it is the decoder. At B4 the
non-decoder per-slot work is ~4-8 ms of a 93 ms frame; the decoder is 35 ms.

---

## 12. Decoder Amdahl floor on VNNI

From the executed-path census of the 1.7B B2 single-CCX run (96 frames) and the code:

| question | answer | label |
|---|---|---|
| accelerated kernels | residual-block convs (in = out ≤ 768 ch): int8 VNNI panels (`qwen_conv1d_int8`, `dpbusd`); pre-transformer projections and ConvT taps: f32 SGEMM sliced on the engine pool; snake: AVX-512 rows via pool | MEASURED (census: 84 % of decoder GMAC VNNI/bf16, 16 % BLAS, 0 fallback) |
| generic | VQ sums, output proj, windowed attention (window 72), RoPE, layer-norms, dwconv, residual adds, final 7-tap conv, clamp: scalar C on the loop thread | MEASURED from code |
| scales with B | nothing algorithmically; per item | MEASURED (17.4 / 23.7 / ~35 ms at B2/B3/B4) |
| bandwidth- vs compute-bound | 303 GMAC / 96 frames ≈ 3.2 GMAC per slot-frame in 8.7 ms on 8 cores ≈ 45 GMAC/s/core ≈ 5 % of Zen5 VNNI peak; decoder weights are small; im2col f32 traffic is the main byte mover (E3: ~1 GB per 32-frame chunk per stream on chunk32) | INFERRED: glue/latency-bound, not bandwidth-bound |
| thread scaling | 8T → 4T: ×1.48; 8T → 16T: ×0.63 per slot; 50-120 pool dispatches per call per item; conv_stack = 92 % of decoder time on every arm | MEASURED |
| does VNNI product mode resolve to per-item fallback | yes: `decoder.mode = per-item-int8-vnni`; `QWEN_DECODER_BATCH=1` reaches `sd_batch_fallback` (sequential per item); the server's "batched speech decoder ON" banner is misleading here | MEASURED from code + dispatch-map |
| decoder batching algorithmic on this ISA | no | MEASURED |
| 1.7B vs 0.6B difference outside the decoder | all of it: 17.4 ms of Talker weight stream; decoder per slot 8.7 vs 7 (same model, noise/dilution) | MEASURED |
| decoder the dominant fixed floor on 0.6B | at B4: decoder 45 %, CP 36 %, Talker 18 % | MEASURED |

Semantic parity (same server behaviour) holds across ISAs; accelerated parity does not:
AMX Design-D + fused residual is a different decoder from this per-item VNNI path, and every
AMX decoder finding (fused residual −0.043 p95, glue-bound wall) transfers as a *direction*
only.

---

## 13. Cache / model replication / prefork

From `main.c`, `qwen_tts_server.c`, `talker.c`, `cp.c`, `sd.c` (MEASURED from code):

- bf16 safetensors: `mmap(PROT_READ, MAP_PRIVATE)` in the parent; page-cache pages shared by
  all workers.
- int8 Talker/CP weights: quantized in the parent before fork into anonymous memory, never
  written after → COW-shared physically. Not duplicated.
- Per-worker private after fork: VNNI row-sum caches (lazy, first use), decoder int8 weight
  cache (lazy, pre-warm triggers it), KV blocks per slot (`B × 28 × kv_max × kvd` bf16 +
  CP KV), decoder stream states and arena scratch, pool stacks. Wave JSON PSS per worker
  2.0-2.3 GB (1.7B) and the box-wide 10.5-11.3 GB for 4 workers + parent are consistent with
  ~1.9 GB shared int8 weights counted proportionally plus ~1.6 GB private per worker
  (INFERRED).
- No VNNI packed layouts (`QWEN_VNNI_PREPACK` unset; rejected on Zen5 2026-09-03).
- No NUMA/first-touch policy in code; irrelevant on this single-node host.
- Cache residency: each worker's L3 (32 MB) holds a slice of whatever it streams; the CP set
  (112 MB) never fits, so 4x8 does not "benefit from replicated hot CP state" — it benefits
  from *not sharing* an L3 with another team's traffic. 2x16 puts one team's CP rows on two
  L3s (56 MB each, still > 32 MB) and its cache-resident phases run at the 40.5 GB/s of the
  two-CCX probe.

Conclusion: replication is not a cost here and cache sharing is not a benefit; the topology
effect is bandwidth-per-CCX plus team width (§5).

---

## 14. Doctor vNext — three ceilings

What the doctor has now (section 8): physics (bandwidth, perfect batching, free decoder),
model (plus a decoder term that is a ×1.5 GUESS on non-AMX), floor (W × 1), and the measured
calibration points. What this review adds as calibration for the `x86_avx512bf16` family
(MEASURED on one CCX):

    Talker(B) = bytes / roof × (1 + 0.04·(B−1))        [not 1.10 per slot: 27.3/26.2 at B2 mix]
    CP(B)     = 18 ms at B1, +3 ms at B≥2, flat         [1.7B and 0.6B alike]
    Decoder   = 8.7 ms × B at 8T; 12.9 × B at 4T; ~5.5 × B at 16T   [per item VNNI]
    width tax = ~12 ms/frame at 16T, ≥ 30 ms at 32T    [rendezvous/idle; from 1x16 and 1x32]
    prefill   = 58 ms per admission, inline

Proposed output, for this host and 1.7B on 4x8 (INFERRED from the numbers above):

| ceiling | per lane | C total | basis |
|---|---|---|---|
| A. physics | B7 | 28 | weights only |
| B. calibrated current engine | B2 GOOD (0.83), B3 hard-gate only (0.97) | **8 GOOD / 12 hard** | 44 + 10.5·B |
| C1. decoder −20 % per item | B3 at 0.90 | 12 GOOD | 44 + 8.8·B |
| C2. decoder halved | B4 at 0.86 | 16 GOOD | 44 + 6.2·B |
| C3. decoder off the critical path (4+4 partition) | B4 at ~0.68, B5 ~0.81 | 16-20 | max(T_step(4c), 12.9·B) |
| C4. CP batching perfect | already the case | 8 | no headroom |
| C5. serial tax removed | B3 at 0.93 | 8-12 | ≤ 1 ms/slot |

Dominant lost capacity between A and B on this host: decoder ~80 %, per-slot region work
~15 %, serial ~5 %. Between the lane model and 2x16/1x32: the width tax. The doctor can
compute B and C1-C3 from the four measured terms; A stays as the bound. It must never print
C as a promise, and its most misleading current number is the 1x32/2x16 model column,
because it has no width tax (§18 Q18).

Also worth fixing in the doctor (not done here): treat `cache-resident < DRAM` on a mask as
a per-mask defect flag; report `T_step(K)` flatness across K as "bandwidth-bound at K threads,
the rest are free for other work".

---

## 15. What is worth trying — ranked by information gain per minute

**Tier 0 — done in this review (seconds each):** counters/logs, single-CCX B1..B4 census for
both models, 4T and 16T probes, CP effective-B audit (census), decoder scaling and class
audit. Remaining Tier-0 items: `QWEN_STAGE_TRACE`+`QWEN_SD_PHASE` re-run of the census on a
fixed-length text (≤ 60 s) to replace the diluted B3/B4 rows.

**Tier 1 — under 5 minutes each:**

| # | experiment | hypothesis | metric | signature if true | signature if false | runtime | decision unlocked |
|---|---|---|---|---|---|---|---|
| T1-a | two servers 1x4@0-3 + 1x4@4-7, 1.7B C2 each | a CCX's bandwidth is not contended by two 4-core lanes | STREAM p95 per server; Talker ms/step | both ≈ 0.97, Talker ≈ 27 ms | ≥ 1.1, Talker ≥ 32 ms | 30 s | build the partitioned lane (§8) or drop it |
| T1-b | 1x1@0 and 1x2@0-1, 0.6B C1, `QWEN_SD_PHASE=1` | decoder on 1-2 cores stays < 80 ms/frame | decoder ms/frame | ≤ 40 / ≤ 25 | > 80 | 30 s | model C (per-stream decoder cores) viability |
| T1-c | 2x16 cap4 vs 4x8 cap2 at C8 with `QWEN_POOL_SPIN` 4096 / 65536 / 256, 1 wave | the 16T width tax is park/wake, not barriers | csw/s, STREAM p95, unattributed ms | csw halves and p95 drops ≥ 0.03 | p95 flat | 2 min | whether the wide shape can be rescued cheaply |
| T1-d | 1x8@0-7 B3 with `QWEN_SD_SGEMM_CENSUS=1 QWEN_SD_PHASE=1` | decoder wall is dispatch/glue (≥ 50 % outside kernels) | SDPHASE split, dispatch count per call | ≥ 40 % in conv_cnext/pre/post + serial | conv kernels ≥ 80 % | 30 s | E (VNNI decoder path) vs B (overlap) priority |
| T1-e | 1x8@0-7 C3 with long/mixed bank, SL-1 layout pinned | lane B3 on long texts is not worse than 0.966 | STREAM p95, prebuffer | ≤ 0.97 | > 1.0 | 1 min | whether the short-bank lane law is optimistic |

**Tier 2 — only after T1 (each ≤ 15 min):** the default-off partitioned lane screen at
C=3/4 per lane; or, if T1-d says glue, one bounded VNNI decoder change (fuse the scalar
residual/dwconv/norm chain or cut dispatches per call) screened on the lane at B3/B4.

Not proposed: 30-minute sweeps, C12+ multi-topology waves, any AMX-only decoder feature,
any admission policy.

---

## 16. Economics

Prices (MEASURED from the AWS pricing API and spot history, us-east-1, 2026-09-08):

| instance | cores | $/h on-demand | $/h spot (min AZ) |
|---|---|---|---|
| c8a.8xlarge (this host) | 32 Zen5 | 1.724 | 0.68 |
| c8a.4xlarge | 16 Zen5 | 0.862 | 0.34 |
| c8i.4xlarge (8c AMX) | 8 GNR | 0.750 | — |
| c8i.8xlarge | 16 GNR | 1.499 | — |
| GCP c4-highcpu-16 (H8), c4-standard-24 (H12) | 8 / 12 AMX | UNKNOWN | — |

$/GOOD-stream-hour, 1.7B (GOOD = full preferred envelope, which on this host is not yet
qualified; screen values used with that caveat):

| endpoint | GOOD streams | basis | $/h | $/GOOD-stream-h |
|---|---|---|---|---|
| H8 AMX 1x8 C2 (qualified) | 2 | QL-1 SOAK | UNKNOWN | UNKNOWN (c8i.4xlarge as a proxy: 0.375) |
| H12 AMX 2x6 C4 (MARGINAL) | 3-4 | CT-5 | UNKNOWN | UNKNOWN |
| T32 today, 2x16 cap4 C8 (screen, 0.84 p95) | 8 | this screen | 1.724 / 0.68 | **0.216 / 0.085** |
| T32 at C12 (hypothetical, lane B3 at ≤ 0.90) | 12 | §14 C1 | 1.724 / 0.68 | 0.144 / 0.057 |
| T32 at C16 (hypothetical, partition or decoder ×0.5) | 16 | §14 C2/C3 | 1.724 / 0.68 | 0.108 / 0.043 |
| T16 c8a.4xlarge 2x8 (predicted C3-C4 by the judge; not measured on the new engine) | 3-4 | judge §11 | 0.862 | 0.287 / 0.216 |

0.6B on T32: C12 at 250 ms (screen) → 0.144 on-demand / 0.057 spot per stream-hour.

Reading: on-demand, C8 on 32 Turin cores already beats the predicted 16-core Turin points and
the 8-core AMX proxy per stream; the 32-core host's economics are decided by whether one lane
reaches B3 GOOD (−33 % per stream) — not by SIMD utilization. Spot prices are not a
production basis and are shown only because the campaign runs on spot.

CTO-facing row for this host (screen-grade, one wave, short bank; NOT qualified):

| instance | model | highest screen-GOOD C | TTFA p95 | STREAM p95 | prebuffer p95 | safe-start p95 | stall@250 | stall@500 | req/s | $/h | $/GOOD-stream-h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c8a.8xlarge | 1.7B (2x16 cap4) | 8 | 210-223 ms | 0.79-0.84 | 189-276 ms | 377-485 ms | 0 % | 0 % | 3.6-3.8 | 1.724 | 0.216 |
| c8a.8xlarge | 0.6B (4x8 cap4) | 12 | 133 ms | 0.72 | 155 ms | 287 ms | 0 % | 0 % | 4.5 | 1.724 | 0.144 |

---

## 17. Existing lessons, checked against the new evidence

All eighteen priors in the brief survive; three are sharpened:

- "extra physical cores give tail headroom even at lower average core-equivalents": the lane
  census shows 7.2-7.5 core-equivalents of 8 at every B with the frame time still growing
  10.5 ms per slot — the spare capacity is not idle cores, it is serialized decoder time.
- "wide pools collapse even when bandwidth arithmetic predicts they win": now measured with
  the physical signature (40.5 GB/s cross-CCX cache-resident rate, csw 20k-90k/s, ~12 ms/frame
  unattributed at 16T).
- "reducing critical-path material work beats orchestration": still the best-supported
  direction (fused residual, SL-1), *and* the one orchestration change the numbers now
  justify testing is precisely the one that removes serialization rather than adding a
  gate (§8).

---

## 18. Required final judgment

1. **Primary limit:** mixed — cache-domain-limited in shape (the CCX is the lane; wide teams
   collapse) and engine-limited inside the lane (per-item decoder serial on the step thread).
   Not memory-limited above B1: the weight stream is fixed at 44 ms and batching is free.
   MEASURED + INFERRED · HIGH.
2. **Why 0.6B reaches C12/C16 and 1.7B stops at C8:** the models differ only in the fixed
   Talker stream (26 vs 9 ms); the marginal slot costs 10.5 vs 8 ms, both dominated by the
   same per-item decoder. With 36 ms of slot budget the 1.7B lane fits 3 slots barely; with
   53 ms the 0.6B lane fits 6. MEASURED · HIGH.
3. **4x8 the natural topology:** yes as the lane for raising B; today 2x16 cap 4 is the better
   C8 operating point for the unchanged engine (0.79-0.84 vs 0.87-0.90). MEASURED · HIGH.
4. **What prevents B2 → B3 on one 1.7B lane:** 5 ms against the 0.90 gate, contained in the
   third slot's decoder call (8.7 ms serial). MEASURED · HIGH.
5. **B3 → B4:** 21 ms against the gate, 13 ms against realtime; the fourth decoder call plus
   inline prefill share. MEASURED · HIGH.
6. **CP batching already working:** yes — one region per frame, one weight traversal per
   step for all slots, 18 → 21 ms flat from B2 to B4. What remains expensive is its fixed
   1.8 GB re-read at the CCX roof, the 718 spin barriers per frame with idle threads during
   per-slot sections, and the 16-step sequential dependency; none scales with B. MEASURED ·
   HIGH.
7. **Decoder the dominant fixed floor:** on 1.7B it is the dominant *marginal* cost (80 % of
   each extra slot) and 26-35 ms of a B3/B4 frame; on 0.6B it is the largest single term at
   B4 (45 %). MEASURED · HIGH.
8. **Time to remove for C12:** 5 ms per frame per lane at the preferred gate (0 at the hard
   gate on the short bank). INFERRED · HIGH.
9. **For C16:** 21 ms per frame per lane at the preferred gate, 13 at the hard gate. INFERRED ·
   HIGH.
10. **Partitioned decoder overlap plausible before coding:** yes on paper (max(53, 52) ms at
    B4), with one unmeasured term (CCX contention) that a 30-second experiment settles.
    INFERRED · MEDIUM.
11. **CPU-native lane/stream ownership worth pursuing:** the lane-per-CCX model already
    holds; the extension worth pursuing is decoders on the lane's spare cores, not per-stream
    cores for everything (8x4 is bandwidth-infeasible for 1.7B). INFERRED · MEDIUM.
12. **Deadline-aware scheduling material?** Not in a lockstep lane (reshuffling; the gate
    already showed parking loses). Material only for ordering work on a separate decoder
    lane. INFERRED · MEDIUM-HIGH.
13. **The ONE measurement:** two 4-core lanes on one CCX (1x4@0-3 + 1x4@4-7, 1.7B, C2 each):
    contention of the CCX bandwidth. 30 s. HIGH.
14. **The ONE implementation experiment after it:** default-off partitioned lane (engine
    pool 4 cores, decoder private team 4 cores, one-chunk bounded async handoff), screened at
    1x8@0-7 C3/C4 against the §4 census. If T1-d says the decoder wall is dispatch/glue, the
    alternative of equal expected value is cutting the VNNI decoder's dispatch count and
    scalar chain (E). MEDIUM.
15. **C12 on 1.7B plausible on this host:** yes at the hard gate now (0.966), yes at the
    preferred gate with 5 ms removed per lane-frame. INFERRED · MEDIUM-HIGH.
16. **C16 plausible:** only with decoder overlap or ~halved decoder per item; not with
    scheduling, not with wider pools. INFERRED · LOW-MEDIUM.
17. **Absolute physical ceiling with a much better engine:** bandwidth alone allows B7 per
    CCX (C28) for 1.7B; with a decoder at even 25 % of peak per item (~2 ms/slot) the lane law
    becomes 44 + 4·B → B9, C36 on weights; the honest engineering ceiling is the partition
    line, C16-C20. SPECULATIVE · LOW.
18. **Most misleading doctor number:** the model column for wide shapes (1x32 C16, 2x16 C12):
    no width tax, decoder term guessed; and the physics column read as capacity. The
    single-CCX rows are the trustworthy ones. MEASURED (by falsification) · HIGH.
19. **Stop testing:** 1x32 and 8x4 for 1.7B; C10+ waves on the current engine; chunk
    sweeps as capacity levers (they trade prebuffer for RTF, measured again here: chunk 4
    +2-3 ms/step, −120 ms prebuffer); `QWEN_DECODER_BATCH` on VNNI as if it batched; cap and
    admission thresholds; global cross-worker batching; warm-vs-cold repeats. HIGH.
20. **Sixty minutes on this Spot box, in order:**
    1. (1 min) T1-a: two 1x4 lanes on CCX 0, 1.7B C2 each — the contention number.
    2. (2 min) fixed-text single-CCX census with `QWEN_STAGE_TRACE=1 QWEN_SD_PHASE=1` at
       C=1,2,3,4 for 1.7B — the undiluted B3/B4 rows and the decoder split (T1-d folded in).
    3. (1 min) T1-b: 0.6B on 1 and 2 cores — decoder single-core cost.
    4. (2 min) T1-c: 2x16 C8 spin-budget A/B — is the width tax cheap to buy back.
    5. (2 min) T1-e: lane B3 on the long/mixed bank with the SL-1 layout pinned.
    6. (10 min) 2x16 cap 4 C8 and 4x8 cap 2 C8 with the pre-registered `vnni-product`
       profile, 3 waves, short + medium — the first citation-grade Turin points.
    7. (remaining) write the numbers into the addendum and the doctor's calibration; do not
       start code on the box.

---

## Adversarial check

**"4x8 is right and the only problem is B3/B4 per worker."** Attacked in §6. It survives with
two amendments: 2x16 cap 4 is the better shape for the *unchanged* engine at C8, and "the
only problem" is specifically the per-item decoder on the lane's critical path plus inline
prefill; nothing else in the lane scales with B.

**"The engine is leaving huge easy performance on the table."** Partly false, partly true.
False: there is no single serialization point wasting half the machine; batching of the
head, CP and Talker is real and nearly free; the weight stream is at the CCX roof; C8 is
within 10-15 % of what the *current* decoder design allows on four lanes. True: the decoder
does its work at ~5 % of VNNI peak with 50-120 pool round trips per call and a scalar chain
on the loop thread, and it is serialized with the step; that is where the gap to C12-C16
lives, and it is neither easy nor speculative — it is measured, bounded (5 ms for C12, 21 ms
for C16 per lane-frame), and has one cheap falsifier before any code.

Better-supported interpretation: **C8 is close to the ceiling of the current 1.7B engine on
this host; C12 is a bounded decoder problem; C16 needs the decoder off the critical path.**
The next move is A/B/E in the brief's list — optimize the lane, by overlapping or shrinking
the decoder — decided by T1-a and T1-d; not C (scheduling), not D (CP batching, already
done), and not F (accept C8) yet.

---

## 19. Amendment after the Tier-1 runs (2026-09-09, §18.20 steps 1-5 executed)

Full numbers: `.work/turin-c8a-32c-fast-screen-20260908.md` §8. What they change above:

1. **§4/§7 lane law, undiluted (one fixed text, STAGE trace):** 1.7B `T(B) ≈ 40 + 13.5·B` ms
   (STREAM p95 .674/.861/.991/1.172), not 44 + 10.5·B. Per slot: decoder 9.7 ms per decoded
   slot-frame (72 %), Talker +1.6, CP +1.8, loop-thread serial 0.05. **B3 deficit at the 0.90
   gate = 7 ms; B4 = 22 ms.** §11's conclusion stands: the serial tax outside the decoder is
   negligible. The decoder-call iterations (wall p95 128/162/205 ms at B2/B3/B4) are the
   prebuffer driver; a per-slot phase stagger of the 8-frame cadence would flatten them at
   the same mean (SPECULATIVE, cheap, distinct from the rejected gate — it delays no work).
2. **§12 decoder anatomy:** 48 % of a decoder call is `res1`, the dilated k=7 int8 VNNI conv1
   inside the residual blocks — one kernel, not "glue". The E path has a named target.
3. **§8 partition, T1-a:** two 4-core lanes on one CCX degrade to 1.28 (from 0.97): a CCX
   carries one weight stream, and it saturates at **2 threads** (1x2@0-1: Talker 26.4 + CP
   20.8). The partition arithmetic becomes stronger (2 step cores + 6 decoder cores; one core
   decodes one stream at 33 ms/frame) but its open term is now L3 pollution of the CP's
   resident rows, not bandwidth. Cheapest falsifier: `tests/decode_quantum_bench` pinned on
   cores 2-7 as decoder-only load while `1x2@0-1` serves B1; signature = CP ms/step (18 → 33
   would kill it).
4. **§5 width tax, T1-c:** `QWEN_POOL_SPIN=65536` on 2x16 C8: STREAM p95 0.893 → 0.808, TOTAL
   0.988 → 0.910, csw 38k → 7.6k/s. About half the wide-team tax is park/wake. Candidate for
   the 2x16 product lane after a SOAK; the 2x16 C8 point itself varies 0.79-0.89 across
   one-wave screens and needs 3-wave cells.
5. **§6/§18 long texts, T1-e:** lane B3 on long texts 0.917, B2 0.792 with SL-1 pinned; the
   short-bank lane law is conservative, and C12 at the preferred gate is within ~2 % on long
   texts.
6. **§18.20 item 6**, pre-registered `vnni-product`, 3 waves, short+medium (fast-screen
   addendum §9): steady state agrees with the provisional profile (4x8 C8 STREAM p95 0.885,
   2x16 C8 0.900, stall@250 0 %), but TTFA p95 is 686-2566 ms and TOTAL 1.1-2.5 because the
   lane pins `QWEN_PREFILL_MATMAT=0` and the no-BF16 policy: the prefill falls back to f32 BLAS
   (~600 ms per admission instead of ~58) on a CPU that has native `avx512_bf16`. The
   pre-registered Turin Phase 4 cannot use that profile as committed; a `vnni-bf16-product`
   lane is the prerequisite. Also visible in §16: the $/GOOD-stream figures for this host
   remain screen-grade until that lane exists.

---

## 20. Contention falsifier executed — final judgment on the intra-CCX pipeline (2026-09-09)

Numbers and design: fast-screen addendum §10-11. In one line each:

- **Falsifier PASS (MEASURED, HIGH):** a decoder-only load on 6 cores of the CCX costs a
  2-core step lane +12 % (CP 21 → 24 ms, Talker 26.5 → 29.3); the CP's L3 residency survives.
- **2 step + 6 decoder: NO-GO (MEASURED, HIGH):** a 2-thread step lane pays +17 ms per slot
  in the regions' per-slot sections; B4 = 98 ms before any decoder.
- **4 step + 4 decoder: GO (INFERRED, MEDIUM-HIGH):** max(T_step(4c) × 1.12, decoder lane) =
  64 ms at B3 (0.80), 72 at B4 (0.90), 79 at B5 (0.99). C12 GOOD and C16 at the gate edge on
  four lanes, from C8 today; TTFA unchanged.
- **Decoder on another CCX: not the next falsifier** — unnecessary after the pass, and
  worse on the evidence (cross-L3 activation traffic, and it steals the other CCX's lane).
- **`vnni-product` TTFA: profile/backend-selection bug** (f32 prefill pinned on a CPU with
  native BF16), excluded from all architecture claims; fix in the profile control plane.
- **Stop benchmarking.** The smallest implementation hypothesis (default-off lane split with
  a one-chunk mailbox per slot, EDF service, engine pool 4 + private decoder team 4, pinned)
  is written in addendum §11 with its single falsifier.
