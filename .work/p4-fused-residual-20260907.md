# P4 fused-residual AMX candidate — 2026-09-07

## Question

Does the existing `QWEN_SD_FUSED_RESIDUAL` path remove a real decoder fixed-cost pass
without changing audio, and does that local saving survive the production ragged-batch
server path?

## Scope and provenance

This is a bounded Tier-A candidate test, not a C4 qualification.  The code was unchanged
by the candidate itself; commit `a536c8b` adds only a cold `QWEN_SD_PHASE` counter for
fused calls.  The AMX binary was built clean from source fingerprint `a536c8b:clean`,
`SIMD=amx`, SHA-256
`fcaf0bfc258e3135cb2d9d9909fc622db70db6e290ee8e4259babc7f78e8fc69`.

The reference was GCP `c4-standard-24`, Xeon Platinum 8581C, one socket/NUMA, twelve
physical CPUs `0-11`, SMT off, `2x6`, engine-owned decoder pool, batch cap 2, 1.7B
INT8, Design-D AMX, warm strip, ragged threshold 2, q8 and synchronous output.  Arms
ran sequentially with the same short diverse bank (`load_texts_en.txt`, `short`), seed
base 2027, three true simultaneous waves at C3 and C4, and no profiler/census.  Receive
coalescing was 0% in every KPI cell.  The shipped source tree has no `.git` metadata, so
the harness's generic dirty field is not authoritative here; the embedded binary
fingerprint is clean.

## Audio and path gates

On the current binary, identical CLI control/treatment runs with the same seed and text
were byte-identical.  `compare_audio.py --min-corr 0.99 --dur-tol 0.05` passed with
`mel_corr=1.00000`, duration `1.28 s` versus `1.28 s`; no process survived teardown.
Earlier multi-text CLI pairs were also byte-identical.  This establishes the tested
numerical/audio contract, not subjective equivalence for every voice or model.

A separate one-wave C4 server diagnostic enabled `QWEN_SD_PHASE=1`; its timing output is
not KPI evidence.  The per-worker log contained 49 `[SDUP]` records, including 11
`path=ragged` records.  Each of those records reported `fused_residual=12`; the same
counter appeared on the per-slot records.  Model load reported 24 persistent Design-D
INT8 B packs (19.3 MB).  Thus the treatment flag was not merely accepted by startup: the
fused branch executed in both server decoder forms.  Phase `unacc` values are nested
timer-accounting artefacts and are not used below.

## Server A/B

All 21 requests per arm completed with errors, rejects and timeouts `0/0/0`.
Percentiles are nearest-rank values from the playback-aware client; all playback values
are client-observed.  `B` is measured effective batch, not client concurrency.

| arm | C | TTFB p50/p95 ms | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p50/p95 ms | safe start p50/p95 ms | max gap p95 ms | stall@250 / @500 | req/s | B | cores |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| fused off | 3 | 1 / 57 | 159 / 166 | 0.782 / 0.832 | 0.836 / 0.886 | 149 / 298 | 310 / 457 | 522 | 11% / 0% | 1.47 | 1.89 | 8.9 |
| fused on | 3 | 1 / 63 | 153 / 166 | 0.667 / 0.820 | 0.698 / 0.875 | 171 / 285 | 324 / 441 | 510 | 11% / 0% | 1.38 | 1.78 | 8.2 |
| fused off | 4 | 54 / 60 | 171 / 179 | 0.778 / 0.831 | 0.835 / 0.905 | 187 / 389 | 359 / 568 | 521 | 17% / 0% | 1.71 | 2.35 | 9.0 |
| fused on | 4 | 53 / 60 | 165 / 170 | 0.764 / 0.788 | 0.826 / 0.862 | 160 / 266 | 328 / 426 | 510 | 0% / 0% | 1.77 | 2.33 | 9.1 |

Fixed-buffer details from the same cells:

- C3: stall@100 was 67% in both arms; stall@500 was 0% in both arms.
- C4: stall@100 was 75% off and 67% on; stall@250 was 17% off and 0% on; stall@500
  was 0% in both arms.
- No errors, rejects, timeouts or coalesced reads occurred.

The C4 result is a clear Tier-A serving signal: STREAM p95 improves `0.831 -> 0.788`,
TOTAL p95 `0.905 -> 0.862`, required prebuffer p95 `389 -> 266 ms`, safe-start p95
`568 -> 426 ms`, and stall@250 goes `17% -> 0%`.  C3 moves in the same direction for
stream p95 and safe-start p95, but request rate and measured cores are lower in this
short sequence.  Therefore this is not evidence that fused residual universally
improves every concurrency or workload.

## C4 mixed-bank SOAK

The treatment then ran alone through the canonical five-minute closed-loop C4 SOAK on
the same host, binary, topology and runtime flags, with the full 21-entry mixed bank
(short, medium, long, conversational and Italian), temperature `0`, 60 s warm-up and
four 60 s analysis windows.  The run produced 119 completed KPI requests over 313.9 s;
all errors, queue rejects and server timeouts were zero, and receive coalescing was 0%.
The binary remained `a536c8b:clean` with the SHA above.  `soak_summary.json` reported
overall latency/resource/drift status `PASS`.

| metric | pooled sustained result |
|---|---:|
| TTFA p50/p95 | 197.7 / 526.2 ms |
| STREAM_RTF p50/p95 | 0.8304 / 0.8933 |
| required prebuffer p50/p95 | 300.3 / 554.5 ms |
| safe play start p50/p95 | 535.3 / 916.8 ms |
| max gap p95 | 966.9 ms |
| stall@100 / @250 / @500 / @1000 | 97.5% / 50.4% / 0.84% / 0% |
| total requests / coalesced reads | 119 / 0% |

The four window STREAM p95 values were `0.8987`, `0.9045`, `0.8617` and `0.8838`.
Thus the pooled hard realtime gate (`p95 < 1`) holds across every window and the
pooled aggregate reaches the preferred `<=0.90` target, but the preferred target is
not a per-window guarantee.  The one @500 ms stall is a single client-observed event;
the zero-buffer prebuffer and @250 ms rates remain stricter playback diagnostics, not
proof of audible starvation with a larger application buffer.  Per-class p95 drift was
not assessable because each class contributed only 5–7 samples per window, below the
20-sample analyzer requirement.

## Verdict

**PROMOTE AS AN ISOLATED, DEFAULT-OFF P4 CANDIDATE; C4 POOLED HARD GATE PASSED.**  The
path is numerically clean in the tested CLI contract, executes in both per-slot and
ragged server forms, and the mixed-bank SOAK confirms `STREAM_RTF p95 < 1` in every
window without a TTFA regression.  Keep `QWEN_SD_FUSED_RESIDUAL` default-off until a
deployment policy explicitly selects it and the remaining per-class/longer playback
coverage is desired.  Do not claim that decoder intercept or AMX coverage is solved:
this path removes only the same-width 1x1 residual-add pass where its Design-D
conditions hold.

## Remaining gates and constraints

- Preserve the separate output buffer and exact fallback for unsupported shapes, non-AMX
  builds and disabled INT8 paths.
- Before a production default change, repeat the quality/audio gate across the established
  bank and run a longer C4 comparison with the same control identity.
- A future fused-residual extension must prove its own quantization/accumulation contract;
  this result does not license fusion of ConvT, depthwise or arbitrary residual blocks.
- The candidate remains unrelated to the rejected persistent scratch-reuse experiment and
  does not justify more ragged threshold or AMX micro-tuning.

## Next action

Use this candidate as the treatment for a longer C4 qualification arm.  If it holds the
playback-aware envelope, retain the flag only if deployment policy explicitly selects it;
otherwise leave the default conservative and move to a separately justified structural
decoder-intercept hypothesis.
