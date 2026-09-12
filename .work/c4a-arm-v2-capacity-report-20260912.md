# Arm v2 c4a 32-core capacity report

Date: 2026-09-12
Source: commit `1308d17`, clean tree
Machine: GCP c4a-highcpu-32, Neoverse-V2, 32 vCPU, topology `4x8`
Workload: OSS Qwen3-TTS 0.6B and 1.7B, speaker `ryan`, English, int8, short text bank

This is a public-safe report. Raw request logs, profiles and audio remain in the ignored
evidence area. No customer model, checkpoint or private path is referenced here.

## Method

The control is `arm-product`. The exploratory all-on profile additionally enabled BF16
pre-up, elastic/split decoder lanes and multislot (`QWEN_SD_BF16_PREUP=1`,
`QWEN_SD_LANE_ELASTIC=1`, `QWEN_SD_LANE_SPLIT=4`, `QWEN_SD_MULTISLOT=2`). It is not a
production promotion.

WAVE means three synchronized waves at each C level. SOAK means six minutes at fixed C,
with 60 seconds warm-up and the normal strict KPI gate. Both used `batch-size=4`,
`max-queue=0`, `queue-timeout-ms=0`, prefork `4x8`; therefore rejects at high C are
intentional fail-fast admission results, not hidden queueing.

Units: TTFB/TTFA and safe-start are milliseconds; RTF is dimensionless; prebuffer is
seconds. Each timing cell is `p50/p95`. SOAK `N/completed` is post-warm-up KPI samples
over total completed requests. `err/rej` is request errors over intentional rejects.
`stall@250/500` is the percentage of accepted KPI requests that stalled against the
250 ms or 500 ms playback threshold. `starved` is the number of WAVE requests with a
non-zero underrun.

## SOAK results

| model | profile | C | gate | N/completed | TTFB ms | TTFA ms | STREAM RTF | TOTAL RTF | prebuffer p95 s | safe-start p95 ms | stall@250/500 | err/rej | failures |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1.7B | control | C6 | PASS | 302/308 | 23.8/67.3 | 72.1/123.8 | 0.61/0.86 | 0.62/0.89 | 0.46 | 612 | 5.3%/2.0% | 0/0 | — |
| 1.7B | control | C8 | FAIL | 315/323 | 29.2/69.3 | 103.2/123.3 | 0.80/1.08 | 0.80/1.08 | 0.77 | 886 | 20.3%/6.3% | 0/0 | TTFB p50; per-class KPI drift |
| 1.7B | control | C12 | PASS | 355/367 | 76.0/491.8 | 155.0/536.4 | 1.00/1.31 | 1.01/1.37 | 2.24 | 2433 | 61.7%/37.7% | 0/0 | — |
| 1.7B | control | C16 | FAIL | 400/415 | 172.7/524.0 | 248.2/592.5 | 1.26/1.67 | 1.29/1.74 | 7.94 | 8352 | 94.5%/81.8% | 256/23 | request errors; per-class KPI drift |
| 1.7B | all-on | C6 | FAIL | 371/377 | 8.6/34.9 | 66.2/102.2 | 0.51/0.57 | 0.52/0.58 | 0.06 | 141 | 0.0%/0.0% | 0/0 | per-class KPI drift |
| 1.7B | all-on | C8 | PASS | 469/477 | 16.3/37.0 | 75.6/102.4 | 0.52/0.57 | 0.53/0.58 | 0.06 | 154 | 0.0%/0.0% | 0/0 | — |
| 1.7B | all-on | C12 | PASS | 577/589 | 23.4/42.9 | 96.2/166.3 | 0.63/0.67 | 0.64/0.69 | 0.11 | 226 | 0.0%/0.0% | 0/0 | — |
| 1.7B | all-on | C16 | FAIL | 594/610 | 22.4/46.0 | 150.0/226.0 | 0.81/0.87 | 0.82/0.91 | 0.22 | 426 | 0.0%/0.0% | 3/17 | request errors; per-class KPI drift |
| 0.6B | control | C16 | FAIL | 299/315 | 75.5/550.6 | 174.0/604.2 | 1.16/1.44 | 1.17/1.47 | 7.55 | 7733 | 94.0%/77.3% | 2/14 | request errors; server request timeouts; TTFB p50; per-class KPI drift |
| 0.6B | control | C20 | FAIL | 289/305 | 81.1/544.0 | 183.6/593.7 | 1.15/1.36 | 1.16/1.41 | 7.22 | 7411 | 94.1%/79.2% | 24974/4646 | request errors; server request timeouts; per-class KPI drift |
| 0.6B | control | C22 | FAIL | 296/311 | 89.9/491.6 | 178.0/527.5 | 1.13/1.31 | 1.15/1.34 | 6.68 | 6931 | 94.3%/75.0% | 32226/7071 | request errors; server request timeouts; TTFB p50; per-class KPI drift |
| 0.6B | control | C24 | FAIL | 304/320 | 81.7/495.5 | 175.8/532.8 | 1.10/1.27 | 1.12/1.31 | 5.67 | 5824 | 92.1%/71.7% | 44926/9829 | request errors; server request timeouts; TTFB p95; TTFA p95; per-class KPI drift |
| 0.6B | all-on | C16 | FAIL | 420/436 | 21.8/79.4 | 162.7/223.9 | 0.80/0.86 | 0.81/0.89 | 0.27 | 480 | 0.0%/0.0% | 0/13 | per-class KPI drift |
| 0.6B | all-on | C20 | FAIL | 409/425 | 22.2/76.2 | 162.6/224.8 | 0.81/0.88 | 0.82/0.91 | 0.27 | 432 | 0.0%/0.0% | 1344/3986 | request errors |
| 0.6B | all-on | C22 | FAIL | 408/424 | 21.0/73.2 | 163.0/224.4 | 0.81/0.88 | 0.82/0.90 | 0.27 | 445 | 0.0%/0.0% | 2015/6283 | request errors; per-class KPI drift |
| 0.6B | all-on | C24 | FAIL | 414/430 | 21.5/77.8 | 161.6/225.6 | 0.81/0.87 | 0.82/0.90 | 0.27 | 438 | 0.0%/0.0% | 5802/10028 | request errors; TTFB p95; per-class KPI drift |

The SOAK gate is stricter than “the process stayed alive”: a row can have no request
errors and still fail because its per-class KPI sample/drift gate is not satisfied.

## WAVE results

| model | profile | C | ok/err/rej | TTFB ms | TTFA ms | STREAM RTF | TOTAL RTF | prebuffer p95 s | starved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.7B | control | C1 | 3/0/0 | 0.4/0.5 | 39.6/39.7 | 0.37/0.37 | 0.37/0.37 | 0.00 | 0 |
| 1.7B | control | C4 | 12/0/0 | 0.9/1.1 | 47.6/49.6 | 0.40/0.42 | 0.40/0.43 | 0.00 | 0 |
| 1.7B | control | C6 | 18/0/0 | 1.2/21.8 | 85.2/92.8 | 0.58/0.83 | 0.61/0.84 | 0.17 | 12 |
| 1.7B | control | C8 | 24/0/0 | 19.7/24.4 | 83.8/87.9 | 0.71/0.93 | 0.73/0.93 | 0.43 | 24 |
| 1.7B | control | C12 | 36/0/0 | 22.2/45.6 | 130.7/143.5 | 0.86/1.14 | 0.88/1.17 | 0.77 | 36 |
| 1.7B | control | C16 | 48/0/0 | 38.2/63.5 | 160.5/168.3 | 1.10/1.48 | 1.13/1.50 | 1.19 | 48 |
| 1.7B | all-on | C1 | 3/0/0 | 0.4/0.5 | 45.2/45.5 | 0.32/0.33 | 0.34/0.34 | 0.00 | 0 |
| 1.7B | all-on | C4 | 12/0/0 | 1.0/1.6 | 54.8/61.1 | 0.35/0.36 | 0.36/0.38 | 0.00 | 0 |
| 1.7B | all-on | C6 | 18/0/0 | 1.6/21.6 | 100.2/110.2 | 0.45/0.51 | 0.47/0.55 | 0.02 | 12 |
| 1.7B | all-on | C8 | 24/0/0 | 19.3/30.6 | 110.9/112.9 | 0.47/0.50 | 0.51/0.54 | 0.02 | 24 |
| 1.7B | all-on | C12 | 36/0/0 | 25.1/57.7 | 158.4/187.5 | 0.62/0.69 | 0.68/0.73 | 0.12 | 34 |
| 1.7B | all-on | C16 | 48/0/0 | 36.1/99.9 | 176.2/230.0 | 0.81/0.91 | 0.88/0.97 | 0.26 | 48 |
| 0.6B | control | C1 | 3/0/0 | 0.5/0.5 | 30.0/30.0 | 0.28/0.28 | 0.28/0.28 | 0.00 | 0 |
| 0.6B | control | C4 | 12/0/0 | 0.8/1.3 | 32.1/33.7 | 0.29/0.29 | 0.29/0.30 | 0.00 | 0 |
| 0.6B | control | C8 | 24/0/0 | 6.9/9.5 | 57.8/59.7 | 0.61/0.80 | 0.62/0.80 | 0.41 | 20 |
| 0.6B | control | C12 | 36/0/0 | 9.3/18.2 | 84.6/88.0 | 0.90/1.22 | 0.90/1.22 | 0.71 | 36 |
| 0.6B | control | C16 | 48/0/0 | 11.2/24.7 | 107.7/112.3 | 0.94/1.25 | 0.95/1.25 | 1.09 | 48 |
| 0.6B | control | C20 | 48/12/12 | 13.2/28.3 | 111.3/115.2 | 1.07/1.40 | 1.07/1.40 | 1.41 | 48 |
| 0.6B | control | C22 | 48/18/18 | 15.0/27.6 | 111.6/114.5 | 1.01/1.35 | 1.02/1.35 | 1.20 | 48 |
| 0.6B | control | C24 | 48/24/24 | 11.9/25.5 | 107.2/111.2 | 0.88/1.23 | 0.89/1.23 | 1.08 | 48 |
| 0.6B | all-on | C1 | 3/0/0 | 0.4/0.5 | 35.8/35.8 | 0.23/0.23 | 0.24/0.24 | 0.00 | 0 |
| 0.6B | all-on | C4 | 12/0/0 | 1.1/1.4 | 39.3/41.8 | 0.25/0.25 | 0.25/0.26 | 0.00 | 0 |
| 0.6B | all-on | C8 | 24/0/0 | 5.8/22.3 | 76.6/79.5 | 0.43/0.44 | 0.44/0.45 | 0.00 | 0 |
| 0.6B | all-on | C12 | 36/0/0 | 8.6/23.2 | 110.1/117.5 | 0.58/0.62 | 0.60/0.63 | 0.08 | 14 |
| 0.6B | all-on | C16 | 48/0/0 | 15.7/37.5 | 112.3/149.6 | 0.76/0.84 | 0.78/0.85 | 0.24 | 48 |
| 0.6B | all-on | C20 | 48/12/12 | 11.6/32.3 | 108.0/156.3 | 0.77/0.84 | 0.79/0.85 | 0.23 | 48 |
| 0.6B | all-on | C22 | 48/18/18 | 12.1/34.1 | 111.3/149.1 | 0.78/0.87 | 0.80/0.89 | 0.23 | 48 |
| 0.6B | all-on | C24 | 48/24/24 | 23.1/36.5 | 114.7/142.7 | 0.77/0.86 | 0.78/0.88 | 0.19 | 48 |

## Quality and conclusions

The paired 1.7B quality smoke produced identical durations, but the all-on audio
correlations were `0.91712`, `0.92542`, `0.90840`, and `0.94147`, all below the `0.98`
acceptance threshold. Therefore the all-on profile remains exploratory even where its
timing is better.

The current cap-4 result is unambiguous:

- 1.7B all-on has strong accepted-request gains versus control at C6/C8/C12 and a
  clean WAVE through C16, but C16 is not a passing sustained qualification point.
- 0.6B all-on removes the severe playback stalls visible in control and is much faster
  for accepted requests, but C20/C22/C24 still hit the cap-4 admission boundary. Their
  WAVE timing cells must not be read as capacity claims because each has rejects.
- The next experiment is the persistent raised-cap campaign: `batch-size=8` per worker,
  `4x8` workers, `max-queue=0`, `queue-timeout-ms=0`, WAVE at C16/20/22/24/28/32/36,
  then six-minute SOAK at C20/22/24/28/32/36 for control and all-on. This isolates the
  admission-cap effect without hiding overload in a queue.

The raised-cap campaign is running separately in tmux on the same box; its results are
not included in this report.
