# Graviton5 Arm v2 full qualification — 2026-09-11

**Task.** Run the committed Arm v2 control and exploratory all-on profiles on a
32-core Graviton5, with the final 4x8 topology, and record serving, dispatch,
quality and resource evidence.

**Question.** Does the clean Arm `arm-product` baseline qualify as the
Graviton5 control point, and can BF16 pre-up plus the Arm lane/multi-slot
features be promoted by the same campaign?

**Known facts.** The candidate is commit `625ba3f` from a clean tree. The
host is a 32-vCPU Neoverse-V3 system with one NUMA node, 48 MiB L3 and SMT
off. The topology screen had already selected 4x8. The control profile keeps
BF16 pre-up and multi-slot off; the temporary all-on arm enables
`QWEN_SD_BF16_PREUP=1`, `QWEN_SD_LANE_ELASTIC=1`, `QWEN_SD_LANE_SPLIT=4` and
`QWEN_SD_MULTISLOT=2`, while retaining RES1_V2/KAI INT8.

**Unknowns.** This campaign does not qualify all-on audio, does not establish
a universal Arm capacity independent of topology/model, and does not replace
the separate Turin regression evidence. The all-on paired-audio result is the
decisive unknown for optional-feature promotion.

**Files/functions inspected.** `configs/perf/arm-product.json`,
`configs/perf/axion-16c-ttfa.json`, the Arm dispatch/preflight and serving
manifests, `tests/serve_soak.py`, `tests/serve_parallel_wave.py`,
`tests/compare_audio.py`, `tools/wav_qc.py`, and the committed binary's caps,
dispatch and self-test reports.

## Evidence

### Identity and dispatch

The binary SHA-256 began with `1f4a1a62fcff7004`, and the embedded source
fingerprint was `625ba3f:clean`. Strict preflight resolved Arm i8mm/BF16,
KleidiAI INT8/Q4/BF16, RES1_V2 and the per-item INT8 DOTPROD decoder. The
all-on manifests additionally showed the decoder lane and multi-slot as
`ACTIVE`, BF16 pre-up enabled, and no dispatch mismatch.

The clean qualification sequence passed `make doctor`, caps/dispatch/self-test,
strict profile checks and the corrected CPU check: 19 passes, 0 failures,
0 warnings, 1 expected skip. The CPU check recorded a clean binary/tree
match, native self-test, fallback self-test, Arm dispatch match, hardware
fingerprint, bandwidth roofs and zero surviving benchmark processes.

All serving runs used the 1.7B model, Ryan/English, INT8 serving, 4 prefork
workers with 8 threads, 4-slot batches, fail-fast admission, and 4x8 CPU
placement. The 0.6B block used the same 4x8 placement and short FAST bank.

### Clean 1.7B C4 SOAK

The control completed 303/303 requests with HTTP 200, zero errors/rejects and
`SOAK RESULT: PASS`. The all-on arm completed 313/313 with HTTP 200, zero
errors/rejects and also passed the serving latency/resource SOAK checks.

| arm | TTFA p50/p95 | STREAM p50/p95 | TOTAL p50/p95 | prebuffer p95 | safe-start p95 | stall @250/@500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 105 / 115 ms | 0.724 / **0.783** | 0.736 / **0.800** | 101 ms | 210 ms | 0 / 0 | 0 / 0 |
| all-on | 118 / 129 ms | 0.692 / **0.758** | 0.706 / **0.776** | 104 ms | 222 ms | 0 / 0 | 0 / 0 |

The all-on arm is faster in sustained RTF, but has a first-audio tail about
14 ms higher at p95 in this SOAK. Both arms are operationally stable.

### 0.6B FAST A/B

| C | control STREAM/TOTAL p95 | all-on STREAM/TOTAL p95 | control TTFA p95 | all-on TTFA p95 | errors/rejects |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.261 / 0.265 | 0.213 / 0.221 | 29 ms | 35 ms | 0 / 0 |
| 4 | 0.440 / 0.449 | 0.407 / 0.415 | 55 ms | 71 ms | 0 / 0 |
| 8 | 0.975 / 0.977 | 0.512 / 0.540 | 90 ms | 116 ms | 0 / 0 |
| 12 | 1.053 / 1.062 | 0.657 / 0.687 | 130 ms | 167 ms | 0 / 0 |

This confirms a real all-on serving gain at C8/C12 on the short bank, with a
TTFA trade-off. It is not sufficient for promotion because the arm changes
arithmetic and decoder scheduling together.

### 1.7B all-on serving ladder

The clean all-on suite passed with zero errors/rejects. Its realistic 4-wave
ladder reported:

| C | TTFA p95 | STREAM p95 | TOTAL p95 | req/s | errors/rejects |
|---:|---:|---:|---:|---:|---:|
| 1 | 47 ms | 0.296 | 0.314 | 1.50 | 0 / 0 |
| 2 | 86 ms | 0.410 | 0.434 | 0.42 | 0 / 0 |
| 4 | 139 ms | 0.609 | 0.614 | 0.40 | 0 / 0 |
| 6 | 194 ms | 0.803 | 0.830 | 0.56 | 0 / 0 |
| 8 | 215 ms | 0.735 | 0.788 | 0.66 | 0 / 0 |

The earlier high-C 1.7B diagnostic waves, useful for capacity shape but not a
qualification contract, were:

| C | control STREAM/TOTAL p95 | all-on STREAM/TOTAL p95 |
|---:|---:|---:|
| 6 | 0.927 / 0.979 | 0.692 / 0.775 |
| 8 | 1.016 / 1.036 | 0.781 / 0.799 |
| 12 | 1.059 / 1.153 | 0.842 / 0.950 |
| 14 | 1.304 / 1.386 | 0.928 / 1.036 |
| 16 | 1.378 / 1.427 | 0.974 / 1.053 |

These C6-C16 rows are synchronized parallel WAVE measurements (three waves per
level), not long SOAKs. The clean paired SOAK was run at C4 only, once with the
control and once with all-on. There was no paired C16 SOAK in this campaign.
C18 belongs only to the earlier exploratory all-on mini-sweep; it was not run
as a control/all-on SOAK pair and is not part of this qualification table.

### Quality gate

The control WAVs were valid and passed structural WAV QC. The all-on paired
audio had equal durations and valid PCM, but its four mel-correlation values
were `0.92255`, `0.88559`, `0.91057` and `0.92467`; the required threshold is
`0.98`. This is a quality-gate failure, not a serving failure. Therefore the
all-on result cannot change production defaults even though its serving SOAK
passed.

### Turin comparison

The existing Turin 4x8 screen reference was about `0.87/0.87` STREAM p95 at
C6/C8. The current Graviton5 control high-C screen was `0.927/1.016`, so the
control is not a demonstrated win over Turin at those two levels. The all-on
screen was `0.692/0.781`, but that comparison is exploratory because its audio
gate failed. The Turin C12 SOAK reference (`0.912` pooled STREAM p95) is not
directly comparable to this Graviton5 C4 SOAK; it is retained as the separate
cross-ISA reference.

## Conclusion

**Baseline Arm control: PROMOTE as a qualified Graviton5 4x8/C4 serving
reference.** It passed the clean dispatch, lifecycle, structural-audio and
serving SOAK gates with zero errors/rejects and zero playback stalls at the
fixed-buffer 250/500 ms checks.

**BF16 pre-up + Arm lane/multi-slot all-on: REJECT for promotion.** The code
path is active and operationally faster, but the paired-audio mel gate fails.
Keep these flags default-off in both Arm JSON profiles. Keep RES1_V2/KAI and
the prepared-state implementation; this campaign found no lifecycle or
dispatch defect in them.

**Cross-ISA claim: KEEP / INCONCLUSIVE.** Arm all-on can beat the old Turin
screen numerically, but the quality failure prevents a product claim. Arm
control is comparable only in the lower concurrency band, not a Turin C6/C8
win.

## Next action

Leave `configs/perf/arm-product.json` and `configs/perf/axion-16c-ttfa.json`
with BF16 pre-up and multi-slot default-off. Keep optional-feature promotion
open for a narrower paired-audio investigation. Before a future release or
new paid capacity ladder, run the planned short Turin regression screen and
record it separately from this Arm qualification; never use the all-on Arm
numbers as a Turin regression result.
