# Task · SQ-2b direct ragged depthwise-convolution slice

## Question

Can the ragged decoder keep ConvNeXt depthwise input/output in the existing global
column-major workset, removing the per-item temporary copy-in/copy-out without
changing the causal tail contract?

## Known facts

The control path allocated a temporary `[channels][item_len]` buffer for every
ragged item, copied the global input into it, called the existing depthwise helper,
then copied the result back.  The candidate `QWEN_SD_DIRECT_DWCONV` arm computes
directly at each item's global offset and updates that item's six-sample tail in
the same order as `cs_dwconv`.

## Unknowns

The direct form removes per-item allocation and copy traffic, but the decoder still
needs one global output buffer for the following MLP/residual stage.  The short
server A/B therefore tests whether this fixed-cost reduction is material at serving
level; it is not a qualification run.

## Files/functions inspected

`qwen_tts_speech_decoder.c` (`rag_dwconv`, `rag_dwconv_direct`,
`conv_decoder_forward_streaming_batch`, `cs_dwconv`), `qwen_tts_kernels.c`,
`docs/feature-flags.md`, `tests/serve_parallel_wave.py`, and the preceding
`.work/p2-sq2-direct-convt-20260907.md` addendum.

## Evidence

### Local gates

After the change, `make clean && make blas -j4` and `./qwen_tts --self-test`
passed on the local native build with zero self-test failures.  `git diff --check`
passed before commit.  The implementation is default-off and keeps the old helper
as the allocation-failure/unsupported-shape fallback.

### Quality and server-path proof

The AMX diagnostic used a GCP `c4-standard-24` Xeon Platinum 8581C, one socket and
one NUMA node, with CPUs `0-11` online and SMT off, 2x6 prefork, engine pool,
batch cap 2, q8, ragged threshold 2, Design-D INT8 and warm strip.  The candidate
was built from the dirty pre-commit snapshot `694a266-dirty:f87ed9fe6bf1`;
the coherent local commit is `cb4be8b`.

Control and direct C2 quality runs used the same 1.7B model, short/medium/long
English bank and seed 5252.  Corresponding returned WAVs were byte-identical.

A separate phase/ragged diagnostic at C4 found 152 per-slot and 14 ragged
decoder entries.  Ragged entries reported `amx=1` and `fallback=0`; cumulative
`[SDUP]` records on the ragged path advanced `direct_dwconv` from 2 through 14.
Thus the new helper was executed by the actual batched server path.  The C2
diagnostic had only per-slot work, so its zero direct count was expected.

### Short server A/B

Both arms used the same dirty candidate binary, 2x6 SMT-off, engine pool, batch
cap 2, q8, threshold 2, two synchronized waves, no profiler, and seed 6262.
Receive coalescing was 4.5% at C2 and 4.9% at C4 for both arms.

| arm | C | STREAM_RTF p50/p95 | TTFA p50/p95 ms | required prebuffer p50/p95 ms | safe start p50/p95 ms | stall@250 / @500 | errors/rejects |
|---|---:|---:|---:|---:|---:|---|---|
| control | 2 | 0.574 / 0.608 | 346 / 353 | 25 / 34 | 363 / 370 | 0% / 0% | 0 / 0 |
| direct | 2 | 0.590 / 0.605 | 342 / 365 | 24 / 38 | 364 / 385 | 0% / 0% | 0 / 0 |
| control | 4 | 0.772 / 0.835 | 457 / 542 | 290 / 381 | 832 / 837 | 50% / 0% | 0 / 0 |
| direct | 4 | 0.798 / 0.826 | 440 / 549 | 343 / 356 | 784 / 905 | 25% / 0% | 0 / 0 |

The C4 sample contains eight requests and is screening evidence only.  Direct
depthwise is neutral-to-slightly favorable in STREAM_RTF p95 and improves the
control's C4 prebuffer p95 in this wave, but direct C2 p50 and C4 safe-start p95
move in the opposite direction.  No statistically defensible serving win is
established.  C4 `TOTAL_RTF` was 0.845/1.077 control versus 0.871/1.071 direct.

## Conclusion

**KEEP, experimental/default-off.**  The slice is mathematically equivalent on
the tested audio outputs and proven on the real ragged AMX server path.  It removes
per-item depthwise temporary allocation/copy work, but the short A/B is not enough
to promote it or call C4 qualified.  Keep it as a composable SQ-2 base while
measuring the next fixed-cost component.

## Next action

Measure startup/admission scaling versus input length on the unchanged q8 control
and the best P2 combination before entering P3.  Then continue SQ-1/SQ-2 only if
the next bounded measurement identifies a material fixed-cost target.
