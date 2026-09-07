# P4 same-pool decoder consumer — 2026-09-07

## Question

Can the existing optional decoder consumer run beside the continuous Talker/CP loop,
submit decoder tiles to the engine-owned pool, and improve playback and throughput
without adding a second decoder team?

## Provenance

The test used clean committed source `3df5ce9` and its AMX binary (SHA-256 prefix
`ae4c45c71bff780c`) on the single-socket/NUMA Xeon Platinum 8581C reference host:
twelve physical CPUs online, SMT off, `2x6`, engine decoder pool, batch cap 2.
The 1.7B INT8 serving controls were Design-D, warm range strip, ragged threshold 2,
q8, and synchronous output.  The harness was `serve_parallel_wave.py`, true
simultaneous wave, short diverse bank, three waves at C3 and C4, no profiler or
census, and zero coalesced reads.  The control ran before the treatment and the
harness terminated each server before the next arm.

Control omitted `QWEN_DECODER_THREAD`.  Treatment added
`QWEN_DECODER_THREAD=1`; both arms kept `QWEN_DECODER_BATCH=1` in the environment.

## Runtime audit

`QWEN_DECODER_THREAD=1` creates one `dec_pool_t` consumer thread with a cloned
decoder context.  Its decoder calls enter `qwen_sd_pool_run`; with
`QWEN_SD_POOL=engine`, the tile work is submitted to the same persistent engine
pool.  The consumer is not a private decoder worker team, but it is an additional
pool submitter and competes for the same CPU, cache and memory-bandwidth budget.

The current implementation also sets the inline `dec_batch` path to zero when
`dec_on` is true.  The consumer has a separate opportunistic grouping queue, but
the observed trace was `group=1` for the sampled calls.  Therefore this is a test
of the actual flag's current behavior, not a proof that every possible same-pool
batched consumer design is impossible.

## Results

All 21 requests per arm completed with errors/rejects `0/0`; this is a short WAVE,
not a qualification.  Values are nearest-rank per-request percentiles from the
playback-aware client; receive fidelity was 0% coalesced reads in every cell.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | required prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall@250 / @500 | cores | context switches/s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| inline control | 3 | 160 / 167 | 0.685 / 0.833 | 0.716 / 0.885 | 166 / 308 | 325 / 469 | 521 | 11% / 0% | 8.0 | 3,464 |
| same-pool consumer | 3 | 884 / 1200 | 0.486 / 1.168 | 0.805 / 1.554 | 344 / 1534 | 1203 / 1677 | 1329 | 0% / 0% | 7.8 | 8,403 |
| inline control | 4 | 169 / 174 | 0.770 / 0.847 | 0.849 / 0.897 | 215 / 425 | 382 / 598 | 511 | 17% / 0% | 9.0 | 3,675 |
| same-pool consumer | 4 | 901 / 1126 | 0.888 / 1.296 | 1.140 / 1.349 | 812 / 1251 | 1111 / 1615 | 1286 | 0% / 0% | 8.8 | 10,700 |

The treatment has no server-level win: at C4 STREAM p95 worsens by 53% relative
to control (`0.847` to `1.296`), TTFA p95 grows by 952 ms, required prebuffer
p95 by 826 ms, and max-gap p95 by 775 ms.  CPU-equivalent utilization does not
increase, while context switches are about 2.9 times higher.  The decoder trace
shows threaded calls with `group=1` and approximately 35–327 ms durations in the
sampled treatment log, consistent with a serialized consumer/queue path rather
than useful overlap.

## Verdict

**KEEP DEFAULT-OFF / REJECT as a serving change.**  The current consumer fails the
P4 acceptance rule: it improves neither playback safety nor throughput and causes
a material TTFA/max-gap regression.  Do not enable `QWEN_DECODER_THREAD` in a
qualification profile.

This result does not justify reopening the old private-team experiment or claiming
that same-pool overlap is mathematically impossible.  A future variant would first
need to preserve the inline decoder's useful batch semantics and expose separate
operation-call, pool-submit and pool-wait counters; that is a new hypothesis, not a
reason to keep this flag active.

## Next action

Move to the remaining P4 structural decoder-cost work only where existing phase
evidence identifies a bounded fixed-cost target.  Keep true intra-call preemption,
static core lanes, BF16/W4 and global Talker/CP ownership out of this checkpoint.
