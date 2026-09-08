# Task · QL-2 GCP C4 highcpu-16 AMX qualification — 2026-09-08

Task · Measure the first lower-cost Emerald Rapids AMX slot with the frozen AMX
product lane and compare 2x4 against 1x8 without changing serving architecture.

Question · Can eight physical AMX cores sustain a playback-safe C4, and which worker
topology gives the best envelope?

Known facts · The requested clean source was `5dbd2ae`. Strict profile preflight
exposed two small operational parity defects before the KPI run: the AMX profiles
encoded the legacy `QWEN_SD_WINDOWED=1` interpretation, and the validator treated
the non-QWEN `OPENBLAS_THREAD_TIMEOUT` process setting as an engine flag. Those were
corrected without runtime/kernel changes in `a3e9ddd`; the target was rebuilt from
that exact clean commit. This is therefore a measured successor to the requested
checkpoint, not a claim against the uncorrected profile.

Unknowns · No 4+1 probe or five-minute SOAK was authorized because neither topology
reached the C4 usable ceiling. Per-KPI `[ITER]`/decoder-group wall distributions
were not enabled in the non-intrusive WAVE and are UNKNOWN. The 1x8 harness does not
export effective-B in its single-worker summary; its log proves maximum in-flight
admission up to four, but not an effective-B distribution. Cost/hour and GOOD
streams/dollar are UNKNOWN because no pricing evidence was attached to this run.

Files/functions inspected · `configs/perf/amx-product.json`, profile preflight and
dispatch tooling, `tests/serve_parallel_wave.py`, topology reports, roof tooling,
and the F1 q4 evidence. The run used the existing playback simulator and did not
enable profiler or census instrumentation.

Evidence · Three-wave true-simultaneous screens were run sequentially for C1–C4 on
both requested topologies. The corrected 1x8 arm was explicitly pinned to online
CPUs `0-7`. All reported KPI cells had zero errors, rejects and timeouts and zero
receive coalescing.

Conclusion · `1x8` is the better eight-core topology and is GOOD through C3. C4 is
NOT GOOD and also misses the exploratory usable ceiling: STREAM_RTF p95 is `0.974`.
`2x4` is GOOD only through C2 and is weaker at C3/C4. This host is a viable C1–C3
AMX deployment candidate for 1.7B under the tested bank, but not a C4 deployment
candidate under the frozen q4 playback contract.

Next action · Keep this slot in the QL-2 comparison as an 8-core AMX/C3 point. Do not
rescue it with new tuning. The next hardware slot should be a non-AMX control (AMD
Turin/VNNI), followed by Axion/Arm once the common-control lane is exercised.

## 1. Source, binary and host identity

| item | measured identity |
|---|---|
| branch | `feature/x86-amx-vnni-oss` |
| target source | `a3e9ddd696a9a76d9ee7390aff1d13c9580863b9`, clean, detached target checkout |
| target binary | `SIMD=amx`; SHA-256 `06b8cba62a2c5de62eaa9c3cec6b45ceeb8513cf65e5220c97d7e44e5e16051` |
| machine | GCP `c4-highcpu-16` |
| CPU | Intel Xeon Platinum 8581C / Emerald Rapids |
| topology | 1 socket, 1 NUMA, 8 physical cores; CPUs `0-7` online and `8-15` offline |
| SMT | `off`, active `0` |
| governor/quota | performance / no CPU quota; `cpu.max=max` |
| shared L3 / RAM | 260 MiB / approximately 31.3 GiB |
| model/workload | Qwen3-TTS 1.7B INT8; English `ryan`; short diverse bank; seed 2027 |
| WAVE | true simultaneous C requests, three waves per C, no profiler/census |

Post-reboot `make cpu-check CPU_CHECK_STRICT=1`, native/fallback self-test, caps,
dispatch map and strict `amx-product` preflight all passed. No server or benchmark
process survived the final run. The preflight resolved:

```text
decoder=ragged-design-d-int8 / ACTIVE
design_d=ACTIVE       fused_residual=ACTIVE       warm_strip=ACTIVE
talker=AMX            CP=AMX                      prefill=AMX
q4=AMX                decoder_batch=1             decoder_pool=engine
QWEN_SD_WINDOWED=0   QWEN_SD_RAG_MIN_PANELS=2    output=synchronous
prefix_cache=1        --max-queue=0
```

The startup log reports 24 persistent INT8 decoder B packs (19.3 MB). The KPI header
also prints the generic compiled `int8 dot` capability as VNNI; that line is not used
as runtime-leaf evidence. The in-process profile preflight and startup dispatch are
the authority, and the log shows AMX Talker tiles at B=3 while B=1/B=2 Talker work
may legitimately use VNNI according to the existing gates. No per-KPI tile census was
enabled by design.

## 2. Post-reboot roofs

The canonical roof tool measured a nearly symmetric split. Values are DRAM GB/s at
the full mask for each scope; these are hardware roofs, not serving throughput.

| mask | threads | read | copy | triad |
|---|---:|---:|---:|---:|
| `0-3` | 4 | 55.54 | 61.60 | 65.88 |
| `4-7` | 4 | 58.52 | 63.13 | 65.68 |
| `0-7` | 8 | 111.24 | 101.81 | 107.57 |

The read asymmetry between worker masks is about 5.4%; triad is effectively equal.
The `make doctor` model used separately measured GEMV inputs of approximately
66.1 GB/s at 4T and 110.3 GB/s at 8T. Those inputs and its rho are labelled
PREDICTED below; they are not a substitute for the serving WAVE.

## 3. Doctor prediction versus measured screen

`make doctor` predicted, for the 1.7B cost model, C4 rho approximately `0.75` for
1x8 and `0.97` for 2x4. It therefore recommended testing both and suggested 1x8.
The model is explicitly calibrated on compute/frame cost and cannot represent the
full playback/admission interaction.

The measured result is materially worse than the 1x8 prediction: C4 STREAM_RTF p95
`0.974`, not `0.75`. The prediction correctly ranked 1x8 above 2x4, but it did not
predict a qualified C4. This is a useful falsification of using doctor rho as a
serving capacity claim.

## 4. AMX product WAVE results

All values are p50/p95 unless stated. Times are milliseconds; RTF is dimensionless;
stall columns are request percentages. `B` is the harness effective worker batch.
`csw/s` is the run-average context-switch rate, not a percentile.

### 2x4, cap 2, masks `0-3` and `4-7`

| C | TTFB | TTFA | STREAM | TOTAL | prebuffer | safe start | max gap | stall @100/@250/@500/@1000 | req/s | B | cores | csw/s | err/rej/to |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 0.4/0.6 | 93.9/97.6 | .664/.674 | .686/.702 | 29/29 | 123/126 | 210 | 0/0/0/0 | .781 | .724 | 3.63 | 1,287 | 0/0/0 |
| 2 | 0.6/1.3 | 103.4/114.4 | .732/.740 | .759/.766 | 59/113 | 163/227 | 269 | 0/0/0/0 | 1.107 | 1.303 | 5.98 | 2,226 | 0/0/0 |
| 3 | 0.8/65.5 | 178.7/193.2 | .874/1.004 | .874/1.076 | 205/264 | 383/449 | 315 | 66.7/0/0/0 | 1.127 | 1.826 | 5.47 | 1,782 | 0/0/0 |
| 4 | 57.6/61.8 | 180.9/185.3 | .971/.983 | 1.016/1.058 | 234/256 | 414/436 | 315 | 100/0/0/0 | 1.429 | 2.415 | 6.09 | 1,879 | 0/0/0 |

Assignments were balanced at C4 (`6/6` requests; worker mean B `1.15/1.26`).

### 1x8, cap 4 screen, explicit mask `0-7`

| C | TTFB | TTFA | STREAM | TOTAL | prebuffer | safe start | max gap | stall @100/@250/@500/@1000 | req/s | B | cores | csw/s | err/rej/to |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|
| 1 | 0.4/1.1 | 69.0/70.9 | .466/.480 | .483/.500 | 0/0 | 69/71 | 153 | 0/0/0/0 | 1.120 | UNKNOWN | 6.77 | 5,103 | 0/0/0 |
| 2 | 1.5/39.8 | 121.9/125.2 | .617/.663 | .657/.713 | 33/40 | 152/164 | 222 | 0/0/0/0 | 1.355 | UNKNOWN | 6.69 | 4,092 | 0/0/0 |
| 3 | 40.4/84.1 | 179.0/184.4 | .751/.784 | .819/.863 | 94/111 | 276/287 | 252 | 0/0/0/0 | 1.407 | UNKNOWN | 6.75 | 4,228 | 0/0/0 |
| 4 | 79.8/118.5 | 232.8/235.6 | .925/.974 | 1.013/1.087 | 211/242 | 447/477 | 330 | 100/0/0/0 | 1.593 | UNKNOWN | 6.69 | 3,906 | 0/0/0 |

The single-worker server log shows maximum admitted in-flight B equal to C at each
screen level, including B4 at C4. The WAVE parser does not publish an effective-B
statistic for this single-worker form, so no average is fabricated. Iteration-wall
and decoder-group-wall p50/p95 by B are UNKNOWN for both topologies because no
`[ITER]` or decoder phase diagnostic was enabled in these KPI arms.

Receive coalescing was `0.0%` in every cell. The zero-buffer diagnostic reports
startup lead/stall events at C2–C4, but the fixed 250/500/1000 ms stall rates remain
zero through C4 for both topologies; the decisive C4 failure is the STREAM_RTF tail
and the resulting lack of margin.

## 5. Highest playback-qualified concurrency

Applying the frozen gates (preferred STREAM p95 `<=0.90`; exploratory usable ceiling
`<=0.95`; fixed @500 zero; prebuffer p95 preferably `<=300 ms`):

| topology | highest GOOD C | reason for stopping |
|---|---:|---|
| 2x4 | C2 | C3 STREAM p95 `1.004`; C4 `0.983` |
| 1x8 | C3 | C4 STREAM p95 `0.974`, above even the `0.95` usable ceiling |

Thus 1x8 wins the topology comparison, but eight physical AMX cores do not support
GOOD C4 under this 1.7B q4 product lane. Neither topology qualified the condition
needed for the 4+1 overload probe, so no fifth-request result and no SOAK are claimed.

## 6. Comparison with the 12-core AMX anchor

The directly comparable historical anchor is the F1 three-wave q4 screen on the
12-physical-core 2x6 AMX reference. It used the same model class, short diverse bank,
seed discipline, q4/fused/Design-D semantics and playback definitions, but a different
source/binary identity; it is an external anchor, not a controlled same-binary A/B.

| host/topology | highest screened C | STREAM p95 | prebuffer p95 | safe-start p95 | max-gap p95 | stall @250/@500 | req/s |
|---|---:|---:|---:|---:|---:|---|---:|
| 12-core 2x6 F1 q4 | C4 candidate | .868 | 201 ms | 362 ms | 344 ms | 0% / 0% | 1.64 |
| 8-core 1x8 q4 | C3 GOOD; C4 failed | .974 at C4 | 242 ms | 477 ms | 330 ms | 0% / 0% | 1.593 |
| 8-core 2x4 q4 | C2 GOOD; C4 failed | .983 at C4 | 256 ms | 436 ms | 315 ms | 0% / 0% | 1.429 |

At C4, 1x8 is `+0.106` absolute STREAM p95, `+41 ms` prebuffer p95 and
`+115 ms` safe-start p95 versus the 12-core anchor. Its request rate is only about
3% lower in this short screen, but the playback safety margin is materially worse.
The 12-core anchor itself was a screen/candidate rather than a new five-minute
qualification in this artifact.

The result also cautions against average utilization as a capacity predictor: 1x8
uses about 6.7 core-equivalents on an 8-core mask yet misses the C4 realtime margin;
the 2x4 C4 run uses about 6.1 and is slightly worse. Memory/lockstep/pacing and tail
behavior matter in addition to average core use.

## 7. Classification and verdict

| item | status | basis |
|---|---|---|
| host identity / SMT / NUMA / quota | MEASURED | post-reboot cpu-check and host files |
| worker-mask roofs | MEASURED | canonical roof tool, masks `0-3`, `4-7`, `0-7` |
| doctor rho and topology recommendation | PREDICTED | cost model; not serving evidence |
| AMX product resolved dispatch | MEASURED DISPATCH | in-process strict profile preflight and startup table |
| per-KPI AMX tile/MAC share | UNKNOWN | census intentionally disabled in KPI arms |
| C1–C4 playback envelope | MEASURED | three-wave WAVE, zero coalescing |
| GOOD concurrency | DERIVED FROM MEASURED KPI | frozen playback gates |
| cost/hour and cost/GOOD stream | UNKNOWN | no pricing evidence in this run |

**Verdict: 1x8 is the better topology, with highest GOOD concurrency C3.**

**C4 verdict: NOT QUALIFIED / NOT USABLE on 8 physical cores.** Both 2x4 and 1x8
miss the required realtime margin, although fixed 250/500 ms stall rates are zero
in this short bank. The cheaper 8-core AMX shape is therefore viable for lower
concurrency or a smaller model, but not as a 1.7B C4 deployment candidate under the
frozen q4 reference.

The slot should remain in the final full-suite report because it is a clean same-ISA
scaling/cost point and it falsifies the doctor-only C4 prediction. The next QL-2
hardware probe should be **AMD/Turin VNNI**, then **GCP Axion/Neoverse-V2**; do not
start another AMX VM before those non-AMX controls are measured.

## 8. Qualification exclusions

An initial 1x8 screen without an explicit CPU mask produced `1W8T_m0-15` despite
SMT being disabled. It is excluded from all tables because it did not satisfy the
real-mask contract. The final 1x8 screen used explicit `0-7` and reported
`1W8T_m0-7`. No other topology or serving knob was changed to rescue the result.
