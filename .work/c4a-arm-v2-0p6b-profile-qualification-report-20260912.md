# c4a 0.6B all-on profile qualification result

Date: 2026-09-12
Source: clean remote tree `1308d17`
Machine: GCP c4a-highcpu-32, Neoverse-V2, 32 vCPU, topology `4x8`
Profile under test: `axion-c4a-highcpu32-0p6b-all-on`

This was the final confirmation campaign for the host/model-scoped all-on candidate.
It used batch cap 8 per worker, `max-queue=0`, `queue-timeout-ms=0`, the OSS 0.6B
model, speaker `ryan`, the same English short bank and seed 42.

## Results

| profile | C | strict SOAK | KPI samples/completed | TTFB ms p50/p95 | TTFA ms p50/p95 | STREAM RTF p50/p95 | TOTAL RTF p50/p95 | prebuffer p95 s | safe-start p95 ms | stall@250/500 | errors/rejects | gate failure |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| all-on | 16 | FAIL | 421/437 | 20.5/76.0 | 162.9/225.4 | 0.81/0.88 | 0.82/0.90 | 0.27 | 482 | 0.5%/0.0% | 0/0 | per-class KPI drift |
| all-on | 20 | FAIL | 408/428 | 28.3/156.4 | 198.0/302.9 | 1.02/1.09 | 1.03/1.13 | 1.53 | 1664 | 64.5%/24.0% | 0/0 | per-class KPI drift |

The C16 result confirms the operational observation: zero errors, rejects or request
timeouts, STREAM_RTF p95 `0.881`, TOTAL_RTF p95 `0.90`, safe-start p95 `482 ms`, and
zero stall rate at 500 ms. It still misses the strict per-class drift gate, so it is a
strong host-local operational candidate, not a canonical qualified point.

C20 is an edge rather than a sub-one tail guarantee: STREAM_RTF p50 is `1.02` and p95
is `1.09`, with `1.53 s` prebuffer p95 and 64.5% stall@250. It remains useful as the
configured soft edge for capacity experiments, but not as the preferred realtime point.

## Paired audio gate

The control/all-on C1 WAVs had identical durations and clean structural WAV QC, but
all four mel comparisons failed the `0.98` threshold:

`0.96306`, `0.95048`, `0.94975`, `0.94558`.

This is a numerical/audio regression under the repository's required gate. Therefore
the profile remains `unqualified` even though its host-local C16 performance is good.
The all-on flags stay enabled only in the explicitly scoped candidate profile; the
generic `arm-product` profile and 1.7B policy are unchanged.

## Decision

- Promote the host-local policy as **C16 preferred / C20 soft edge** for exploratory
  0.6B operation on this c4a 32-core machine.
- Keep the JSON qualification status `unqualified` until the numerical/audio delta is
  either fixed or explicitly accepted by a separate quality decision.
- Keep the 1.7B guidance at C11 conservative / C12 maximum measured on this host.
