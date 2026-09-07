# P2 / SQ-2c — direct warm decoder input

Date: 2026-09-07
Branch: `feature/x86-amx-vnni-oss`
Commits: `f3637dd`, `86cedfa`

## Question

Can the warm Design-D INT8 causal convolution consume the request-local causal tail
and the new input as two read-only sources, avoiding the per-call fp32
`[tail | input]` materialisation without changing decoder semantics?

## Implementation

`QWEN_SD_DIRECT_INPUT=1` is default-off.  The new split range API keeps the existing
Design-D persistent B pack, quantisation, AMX worker decomposition, tail update and
fallback.  `86cedfa` hoists prefix/suffix interval selection out of the inner
channel/tap loop.  Normal, portable and non-AMX paths are unchanged.

## Gates

- Local `make clean && make blas -j4`: pass.
- Local `--self-test`: `0` failures.
- Local flag registry: `192` read / `192` declared.
- `git diff --check`: pass.
- GCP clean AMX build: pass; binary SHA-256
  `912d0f9086bea6953d72ad66f977a8d0748cb3c02ab42a9abacbb02c9f7891ea`.
- GCP: Xeon Platinum 8581C, c4-standard-24, SMT `off`, active `0`, online `0-11`,
  2x6 prefork, AMX INT8/BF16 active.
- CLI control/feature WAV: byte-identical (`265004` bytes) on `f3637dd`.
- Post-optimisation server C1 quality control/feature WAV: byte-identical
  (`341804` bytes) on `86cedfa`.

## Mechanism proof

Phase-only C4 diagnostics on the same host/workload showed:

- control: warm `SDRES1 ext` typically `4.4–4.9 ms`, `split_input_calls=0`;
- feature: warm `SDRES1 ext` typically `0.02–0.04 ms`, `split_input_calls=12`;
- feature retained `discarded=0.0%` and the same persistent decoder packs
  (`24`, approximately `19.3 MB`).

The feature therefore executes the intended server warm path and removes the
materialisation.  It does not imply that the complete decoder call is faster: the
split source traversal and its memory access pattern are part of total cost.

## Serving A/B (FAST, no profiler, true simultaneous wave)

Host/build for all rows: `86cedfa`, binary prefix `912d0f9086bea695`, 2x6, SMT off,
engine pool, Design-D INT8, stream strip on, ragged threshold 2, decode chunk 8,
batch cap 2, 1.7B, `load_texts_en.txt`, coalesced reads about 4–5%.

First 2-wave C2/C4 screen:

| arm | C | TTFA p50/p95 (ms) | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p95 (ms) | safe start p95 (ms) | stall @250/@500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 2 | 116.6 / 351.9 | 0.590 / 0.599 | 0.605 / 0.613 | 38.7 | 377.5 | 0% / 0% | 0 / 0 |
| direct | 2 | 116.8 / 355.1 | 0.575 / 0.576 | 0.590 / 0.591 | 29.6 | 377.3 | 0% / 0% | 0 / 0 |
| control | 4 | 428.7 / 564.1 | 0.775 / 0.798 | 0.827 / 0.884 | 346.3 | 784.0 | 50% / 0% | 0 / 0 |
| direct | 4 | 428.4 / 494.6 | 0.768 / 0.804 | 0.817 / 0.916 | 314.9 | 780.6 | 62.5% / 0% | 0 / 0 |

Follow-up C4, 3 waves / 12 requests per arm:

| arm | TTFA p50/p95 (ms) | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p95 (ms) | safe start p95 (ms) | stall @250/@500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 429 / 548 | 0.762 / 0.808 | 0.806 / 0.980 | 314 | 771 | 0% / 0% | 0 / 0 |
| direct | 425 / 602 | 0.794 / 0.816 | 0.840 / 0.930 | 439 | 939 | 67% / 0% | 0 / 0 |

The three-wave result does not support promotion.  The lower-level saving is real,
but the end-to-end serving effect is neutral-to-negative and cadence is worse in the
final paired run.  Keep the flag and code only as a documented, independently
disable-able experiment; do not enable it in the serving control or spend another
INT8 tuning loop on this exact split traversal.

## Decision

Status: **implemented, validated, not promoted**.  Next P2 work should target the
next measured fixed cost (allocation/rendezvous or another preparation pass), with a
fresh control-vs-feature server A/B.  Do not infer a general rejection of persistent
packed weights or Design-D from this result.
