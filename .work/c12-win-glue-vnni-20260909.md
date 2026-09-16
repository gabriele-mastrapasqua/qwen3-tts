# C12-WIN glue and VNNI preparation falsifiers

Task · C12-WIN-2b

Question · After the BF16 pre-up and ConvT one-GEMM falsifiers, do the smallest
remaining VNNI preparation changes remove enough decoder residency to justify a
server A/B?

Known facts · The frozen Turin product is VNNI per-item INT8 decoding with
`QWEN_SD_RES1_V2=1`, q4 streaming units, and the DL-2 lane. BF16 pre-up is
default-off because its paired same-generation audio gate measured
`mel_corr=.97890`, below the `.98` gate. The ConvT one-GEMM experiment passed
exact decoder parity but was slower by 14–32% across B1–B4 and chunks 1–8.

Unknowns · These microbenchmarks do not establish a new serving point, do not
measure a changed CP-overlap tax, and do not prove that a different layout would
be beneficial on AMX or Arm.

Files/functions inspected · `qwen_tts_speech_decoder.c`, `qwen_tts_kernels.c`,
`qwen_tts_kernels.h`, `qwen_tts.c`, the VNNI decoder benchmark path, and the
current Turin product profile.

Evidence ·

* Allocation-only: replacing the known fully-overwritten decoder temporary
  `calloc` allocations with allocation-only temporaries produced no useful q4
  improvement at B1–B4 in the pinned Turin microbench. No server A/B was run.
  The change was reverted.
* VNNI RES1_V2 split-input: a temporary direct `tail` + `new` implementation
  passed the continuation/self-test cases with exact suffix parity (`max_abs=0`).
  It nevertheless lost to the existing contiguous `[tail|new]` V2 path. Final
  optimized split timings (milliseconds, five warm repetitions, q4/q8 chunks)
  were:

  | B | control q4 | split q4 | delta | control q8 | split q8 | delta |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 72.64 | 73.56 | +1.3% | 138.38 | 146.44 | +5.8% |
  | 2 | 145.58 | 148.50 | +2.0% | 276.89 | 290.00 | +4.7% |
  | 3 | 218.76 | 220.40 | +0.7% | 416.88 | 441.60 | +5.9% |
  | 4 | 291.92 | 292.80 | +0.3% | 555.98 | 586.80 | +5.5% |

  The split path was reverted; no server A/B was justified.
* The proposed `QWEN_STREAM_FIRST_RAMP=14` comparison was not executed. Its
  temporary flag and implementation were removed before completion, so there is
  no evidence to promote or reject that policy.

Conclusion · The allocation-only and split-input mechanisms are REJECTED for
the current Turin VNNI path. ConvT one-GEMM and BF16 pre-up remain separately
recorded as rejected/default-off. No losing experiment is active in the local
runtime tree. The C12-WIN ladder is closed for this work interval without a
qualified C12 win; the existing frozen profile remains the control.

Next action · Reopen only with an explicitly scoped new C12 experiment. Do not
infer a gain from these preparation ideas, and do not enter the deferred
DECODER-XISA track as part of this checkpoint.
