# Arm v2 0.6B raised-cap report

Date: 2026-09-12
Source: commit `1308d17`, clean tree
Machine: GCP c4a-highcpu-32, Neoverse-V2, 32 vCPU, topology `4x8`
Workload: OSS Qwen3-TTS 0.6B, speaker `ryan`, English, int8, short text bank

This is a public-safe report. Raw logs, profiles and audio remain in the ignored
evidence area. No customer model, checkpoint or private path is referenced here.

## Method

The campaign compared `arm-product` control with the exploratory all-on Arm profile
(BF16 pre-up, elastic/split lanes and multislot). It used `batch-size=8` per worker,
four workers with eight threads each, `max-queue=0` and `queue-timeout-ms=0`. Thus the
raised cap tests admission capacity without hiding overload in a queue.

WAVE used three synchronized waves per C. SOAK used six minutes at fixed C with 60 s
warm-up and the strict KPI gate. Timing cells are `p50/p95`; TTFB, TTFA and safe-start
are milliseconds, RTF is dimensionless and prebuffer is seconds. `err/rej` means request
errors/intentional rejects. SOAK stall cells are the percentage of accepted KPI samples
with a stall at a 250 ms or 500 ms playback buffer.

## SOAK results

| profile | C | gate | N/completed | TTFB ms | TTFA ms | STREAM RTF | TOTAL RTF | prebuffer p95 s | safe-start p95 ms | stall@250/500 | err/rej | failures |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | C20 | FAIL | 324/345 | 138.5/545.2 | 225.5/610.8 | 1.34/1.67 | 1.36/1.70 | 14.36 | 14474 | 99.4%/94.8% | 0/0 | server request timeouts; per-class KPI drift |
| control | C22 | FAIL | 317/340 | 153.7/523.9 | 252.4/619.8 | 1.52/1.92 | 1.55/1.99 | 20.94 | 21015 | 99.7%/99.4% | 0/0 | server request timeouts; TTFB p50; per-class KPI drift |
| control | C24 | FAIL | 329/354 | 157.8/601.5 | 261.2/650.3 | 1.64/1.92 | 1.66/1.98 | 23.15 | 23454 | 100.0%/98.5% | 0/0 | server request timeouts; TTFB p95; per-class KPI drift |
| control | C28 | FAIL | 361/390 | 165.1/631.2 | 294.9/710.9 | 1.79/2.10 | 1.80/2.16 | 27.39 | 27692 | 100.0%/99.7% | 0/0 | server request timeouts; per-class KPI drift |
| control | C32 | FAIL | 374/409 | 154.4/685.1 | 333.6/758.0 | 1.95/2.38 | 1.98/2.44 | 30.77 | 31182 | 100.0%/100.0% | 0/17 | server request timeouts; TTFB p50; per-class KPI drift |
| control | C36 | FAIL | 372/404 | 108.7/586.4 | 258.3/668.8 | 2.01/2.32 | 2.02/2.39 | 30.80 | 31076 | 100.0%/100.0% | 25163/4598 | request errors; server request timeouts; TTFB p50; per-class KPI drift |
| all-on | C20 | FAIL | 415/435 | 25.4/152.2 | 199.6/300.3 | 1.02/1.10 | 1.03/1.13 | 1.51 | 1676 | 55.4%/16.9% | 0/0 | TTFB p95; per-class KPI drift |
| all-on | C22 | FAIL | 414/436 | 36.9/161.0 | 220.4/337.5 | 1.09/1.30 | 1.15/1.34 | 7.60 | 7777 | 81.9%/55.3% | 0/0 | server request timeouts; TTFB p50; per-class KPI drift |
| all-on | C24 | FAIL | 428/453 | 36.1/159.3 | 220.4/343.3 | 1.25/1.32 | 1.26/1.37 | 10.46 | 10912 | 100.0%/91.1% | 0/0 | server request timeouts; TTFB p95; per-class KPI drift |
| all-on | C28 | FAIL | 425/454 | 48.7/215.3 | 227.8/417.0 | 1.45/1.54 | 1.47/1.58 | 17.71 | 17914 | 100.0%/100.0% | 0/0 | server request timeouts; per-class KPI drift |
| all-on | C32 | FAIL | 431/464 | 75.0/245.9 | 283.1/453.3 | 1.67/1.78 | 1.69/1.84 | 23.24 | 23480 | 100.0%/100.0% | 6/22 | request errors; server request timeouts; per-class KPI drift |
| all-on | C36 | FAIL | 432/464 | 55.5/220.7 | 282.1/451.0 | 1.68/1.79 | 1.69/1.86 | 23.16 | 23516 | 100.0%/99.5% | 1315/4463 | request errors; server request timeouts |

No raised-cap SOAK row passes the strict gate. The all-on C20 row is the best
exploratory sustained point because it has zero request errors/rejects/timeouts, but it
still fails the KPI gate and is not realtime-safe: TOTAL_RTF p95 is 1.13, prebuffer p95
is 1.51 s and 55.4% of requests stall at the 250 ms threshold.

## WAVE results

| profile | C | ok/err/rej | TTFB ms | TTFA ms | STREAM RTF | TOTAL RTF | prebuffer p95 s | starved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | C16 | 48/0/0 | 14.2/27.9 | 113.8/119.2 | 0.95/1.13 | 0.96/1.14 | 0.75 | 48 |
| control | C20 | 60/0/0 | 12.3/27.6 | 136.5/143.8 | 1.14/1.35 | 1.16/1.36 | 1.43 | 60 |
| control | C22 | 66/0/0 | 18.5/38.3 | 158.1/168.9 | 1.30/1.76 | 1.31/1.77 | 2.36 | 66 |
| control | C24 | 72/0/0 | 20.3/41.6 | 163.4/176.1 | 1.29/1.57 | 1.32/1.58 | 2.04 | 72 |
| control | C28 | 84/0/0 | 22.4/47.2 | 187.0/196.4 | 1.29/1.60 | 1.31/1.62 | 2.23 | 84 |
| control | C32 | 96/0/0 | 28.6/50.2 | 208.4/216.7 | 1.52/1.75 | 1.55/1.78 | 2.45 | 96 |
| control | C36 | 96/12/12 | 30.2/54.5 | 214.7/222.9 | 1.55/1.95 | 1.59/1.97 | 3.07 | 96 |
| all-on | C16 | 48/0/0 | 10.8/31.9 | 125.5/157.3 | 0.80/0.84 | 0.82/0.86 | 0.23 | 48 |
| all-on | C20 | 60/0/0 | 16.7/36.8 | 125.1/183.6 | 0.98/1.03 | 1.01/1.05 | 0.47 | 60 |
| all-on | C22 | 66/0/0 | 17.9/39.7 | 139.6/226.3 | 1.01/1.27 | 1.03/1.28 | 1.07 | 66 |
| all-on | C24 | 72/0/0 | 22.7/40.8 | 141.3/216.5 | 1.17/1.27 | 1.20/1.29 | 1.04 | 72 |
| all-on | C28 | 84/0/0 | 24.2/42.7 | 151.0/249.9 | 1.38/1.47 | 1.41/1.49 | 1.55 | 84 |
| all-on | C32 | 96/0/0 | 26.2/61.2 | 186.0/290.6 | 1.58/1.69 | 1.61/1.70 | 2.21 | 96 |
| all-on | C36 | 96/12/12 | 26.2/62.5 | 188.4/276.8 | 1.55/1.67 | 1.59/1.67 | 2.19 | 96 |

## Decision

- **Admission boundary:** C32 is the highest tested WAVE level with zero rejects for
  both profiles. C36 is beyond the nominal `4 workers × batch-cap 8` envelope and
  rejects 12 requests.
- **Realtime sustained boundary:** no raised-cap C20+ point qualifies. All-on C20 is
  the best exploratory candidate, but it fails the playback/KPI gate. The raised cap
  increases admission capacity; it does not create a product-quality C20+ stream.
- **Promotion:** no JSON default changes. BF16 pre-up, lane changes and multislot stay
  exploratory/off for qualification because the paired all-on quality gate already
  failed and this raised-cap campaign did not add a new audio gate.
- **Next use:** treat C32 as an Arm capacity/admission candidate only. Graviton5 still
  needs same-host, per-model SOAK evidence; this c4a result cannot replace it.
