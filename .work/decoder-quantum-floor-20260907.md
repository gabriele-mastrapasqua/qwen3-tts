# Decoder quantum floor — 2026-09-07

## Scope and provenance

Tier-A falsifier only; no runtime change. Four sequential runs used the same transferred
`78d0426` AMX binary (`a9e0435fa25edbdb`), GCP `c4-standard-24`, Xeon Platinum 8581C,
SMT off, CPUs `0-11`, `2x6`, engine pool, batch cap 2, Design-D INT8, ragged threshold 2,
1.7B English `load_texts_en.txt`, true simultaneous wave, one wave at C3 and C4, no
profiler. Output transport was the synchronous control (`QWEN_SERVER_ASYNC_OUTPUT=0`).
The tar snapshot has no `.git`; the local commit and binary hash are the identity record.

## Results

All arms completed with errors/rejects `0/0`. Percentiles are the harness's per-request
nearest-rank values; receive marks were client-observed and coalesced-read share stayed
small. This is not a multi-wave qualification.

| quantum | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | required prebuffer p95 ms | safe start p95 ms | max gap p95 ms | stall@250 | stall@500 | coalesced |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 433/434 | .808/.930 | .826/.980 | 164 | 597 | 279 | 0% | 0% | .8% |
| 1 | 4 | 424/424 | .962/1.005 | .992/1.053 | 224 | 535 | 273 | 0% | 0% | 1.0% |
| 2 | 3 | 430/434 | .752/.882 | .770/.933 | 195 | 625 | 311 | 0% | 0% | 1.0% |
| 2 | 4 | 424/424 | .863/.892 | .894/.942 | 149 | 568 | 300 | 0% | 0% | 1.8% |
| 4 | 3 | 435/437 | .697/.807 | .716/.860 | 167 | 603 | 285 | 0% | 0% | 2.8% |
| 4 | 4 | 424/425 | .836/.862 | .864/.893 | 277 | 683 | 366 | 0% | 0% | 3.3% |
| 8 | 3 | 432/435 | .651/.771 | .670/.823 | 348 | 783 | 523 | 67% | 0% | 3.2% |
| 8 | 4 | 428/428 | .793/.817 | .824/.850 | 333 | 606 | 602 | 50% | 0% | 5.6% |

Other observed C4 controls were effective batch `1.78/1.80/1.80/1.81` for q1/q2/q4/q8
and roughly `6.4-6.7` core-equivalents. The number of received chunks was 81/43/24/15
at C4 respectively, confirming that the knob changes delivery quantum rather than only
renaming a path.

## Interpretation

* q1 is rejected as a production minimum on this host: C4 STREAM_RTF p95 crossed 1.0
  and TTFA remained about 424 ms despite the much higher call count.
* q2 is the strongest cadence candidate in this screen: C4 prebuffer p95 149 ms,
  safe-play-start p95 568 ms, zero fixed-buffer stalls at 250/500 ms, and STREAM_RTF
  p95 .892. It still needs a multi-wave/quality check.
* q4 is a plausible efficiency/cadence compromise: STREAM_RTF p95 .862, but prebuffer
  p95 277 ms and safe-play-start p95 683 ms.
* q8 remains the RTF control (`.817` p95 at C4) but has materially worse cadence:
  prebuffer p95 333 ms, max-gap p95 602 ms, and non-zero @250 ms stalls in this wave.

The experiment falsifies neither the need for lead-aware scheduling nor the value of
small complete decoder calls. It does show that the minimum efficient quantum is not q1,
and that a future lead policy should be allowed to choose q2/q4 under low lead while
retaining q8 only when the stream has enough slack. No quantum is promoted as the global
default from this one-wave screen.

## Next action

Keep q8 as the existing upper control and use the current binary/config path for future
bounded lead/credit experiments. Do not combine a quantum change with topology, batch-cap,
ragged-threshold or backend changes.
