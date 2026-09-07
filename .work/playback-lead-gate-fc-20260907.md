# Task · F-C minimal playback-lead gate — 2026-09-07

## Question

Can a minimal per-stream credit gate improve playback cadence by suppressing a
stream's next complete Talker/CP frame once its estimated delivered-audio lead
is above a bounded target, without losing serving throughput?

## Known facts

- The implementation is `04f00d3`, behind `QWEN_STREAM_LEAD_GATE=1`; the default
  remains off.  The hook runs only at an existing complete-frame boundary and
  never preempts a decoder call.
- The server records successful PCM samples and a monotonic first-audio time per
  request.  The first frame is always eligible.  The 250 ms target is the
  experimental default and was the only treatment value tested.
- The source was a clean committed archive of `04f00d3`; the AMX binary SHA-256
  prefix was `8358ca9c520e0986`.  The archive has no `.git`, so the harness's
  in-process source field was `unknown`; the explicit source commit and binary
  hash are the run identity.

## Unknowns

- This screen does not decide whether lead ordering without suppression, or a
  lead-aware quantum policy that preserves useful cohorts, can work.
- The lead is an output-ready estimate.  With the optional detached writer it
  would mean enqueued rather than socket-consumed PCM; that mode was not used.
- No new AMX census was enabled.  The run inherited the previously validated
  Design-D path; the startup flags prove the requested configuration, not a new
  whole-request AMX wall share.

## Files/functions inspected

`qwen_tts.h` (`qwen_batch_sink_t`), `qwen_tts.c`
(`qwen_tts_serve_continuous` frame eligibility), `qwen_tts_server.c`
(`batch_job_t`, PCM-ready accounting, `sink_step_allowed`, scheduler wiring),
`docs/feature-flags.md`, and `tests/serve_parallel_wave.py`.

## Evidence

Both arms used the same GCP `c4-standard-24` Xeon Platinum 8581C, one socket/
NUMA, 12 physical cores, SMT off, CPUs `0-11`, `2x6` prefork, engine pool,
batch cap 2, 1.7B INT8, q8, ragged threshold 2, Design-D INT8, synchronous
output, `tests/load_texts_en.txt`, true simultaneous waves, two waves at C3/C4,
and no profiler/census.  The control ran first; the treatment changed only
`QWEN_STREAM_LEAD_GATE=1,QWEN_STREAM_LEAD_TARGET_MS=250`.  Both runs had zero
errors, rejects and timeouts.  Receive coalescing was 6.1% at C3 and 5.1% at
C4 in both arms.

| arm | C | TTFA p50/p95 ms | STREAM_RTF p50/p95 | TOTAL_RTF p50/p95 | prebuffer p95 ms | safe start p95 ms | max gap p95 ms | stall@250 | stall@500 | core-equivalent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 3 | 235/436 | .615/.769 | .645/.822 | 344 | 780 | 521 | 33% | 0% | 6.2 |
| lead gate 250 ms | 3 | 235/436 | .960/.986 | .971/1.011 | 340 | 775 | 691 | 33% | 0% | 4.1 |
| control | 4 | 447/499 | .780/.838 | .805/.939 | 410 | 858 | 594 | 25% | 0% | 7.3 |
| lead gate 250 ms | 4 | 446/495 | .975/.986 | 1.005/1.018 | 359 | 805 | 732 | 25% | 0% | 5.2 |

The control effective batch was 1.52 at C3 and 1.95 at C4; the treatment was
1.48 and 1.90.  The treatment's two worker logs reported 49,500 non-first-step
lead checks and 47,423 suppressions (95.8%).  It therefore parked most frame
opportunities rather than merely changing delivery grouping.  The modest
prebuffer movement did not translate into lower fixed-buffer stall rates, while
STREAM_RTF p95 regressed by 217 points at C3 and 148 points at C4 relative to
this paired screen.  TTFA was effectively unchanged.

The server log showed `QWEN_SD_AMX_D=1`, `QWEN_SD_POOL=engine`, q8 and the
threshold-2 controls in both arms; the treatment additionally showed
`QWEN_STREAM_LEAD_GATE=1` and `QWEN_STREAM_LEAD_TARGET_MS=250`.  This is valid
path/configuration evidence, not a fresh kernel-census claim.

## Conclusion

**REJECT as a serving policy for this realization; KEEP the implementation
default-off as a bounded diagnostic.**  At the measured worker batch sizes, a
hard per-stream audio-lead gate removes useful Talker/CP work and reduces CPU
utilization, but does not improve client-observed cadence enough to offset the
throughput loss.  This falsifies “suppress every stream above a 250 ms target”
for the current q8/B≤2 server.  It does not justify adding EDF ordering or a
more complex lead scheduler without a different, explicitly testable mechanism.

## Next action

Do not enable the flag or make it the basis for LS-2.  Preserve q8 as the
control and the already measured q2/q4 floor candidates.  The next architectural
slice must either reduce a measured fixed cost (so smaller complete calls remain
affordable) or address the known long-input/admission term; it must not be
another target-value sweep of this hard gate.
