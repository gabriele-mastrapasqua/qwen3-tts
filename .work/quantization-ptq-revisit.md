# Deferred research — calibration-aware PTQ for prefill and weight storage

**Status.** Backlog note only, MEDIUM/LOW. No code, no cloud run, no dependency change.
Start only after the current C12-WIN items and the report qualification work are closed.

## 1. WHY THIS IS OPEN AND NOT SETTLED

The production path keeps **prefill in BF16 on purpose**, as a quality safety margin, while
Talker/CP decode runs INT8. Two earlier attempts to lower that precision were rejected:

| attempt | what was rejected | what was NOT established |
|---|---|---|
| straightforward INT8 prefill | **the implementation**, for quality sensitivity | that INT8 prefill is architecturally unusable |
| straightforward INT4/Q4 weights | **the implementation**, for pronunciation and speaker-character drift | that 4-bit weight storage is architecturally unusable |

Both used the engine's direct conversion: per-tensor/per-block scales chosen analytically,
no calibration data, no optimization of the rounding decision. The audio stayed structurally
valid — correct length, no artifacts, passing waveform-level checks — and still drifted
audibly in pronunciation and speaker character. That failure mode is precisely the one a waveform or
duration gate does not catch, which is why it was found late and by ear.

**The distinction this note exists to preserve:** those results falsify **two specific
implementations**, not the direction. Recording them as "INT8 prefill is disproven" or
"INT4 is disproven" would be a category error, and would close a door that the evidence
never closed. Anyone reading this later should treat the earlier verdicts as
implementation-scoped.

## 2. HYPOTHESIS

Calibration- and optimization-aware post-training quantization (AutoRound-style or
equivalent) chooses rounding to minimize an objective measured on real activations, rather
than rounding to nearest under an analytic scale. Where the naive scheme spends its error
budget uniformly, a calibrated one can spend it where the model is insensitive. The open
question is whether the quality loss that killed the earlier attempts was inherent to the
bit width or an artifact of how the rounding was chosen.

## 3. TRACKS

1. **Calibrated INT8 / W8A8 prefill** that preserves production pronunciation and speaker-character
   behaviour, against the current BF16-prefill baseline.
2. **Quality-optimized Q4/INT4 or mixed precision** for suitable Talker/CP/prefill regions,
   rather than one bit width everywhere.
3. **Offline calibration only.** The rounding/scale search runs outside the engine; the C
   runtime consumes packed weights and scales and nothing else. No training machinery, no
   new runtime dependency, no calibration step at serve time.
4. **Re-test the premise against the V2 dataflow.** The earlier verdicts predate the V2
   kernels. Whether the newer dataflow makes low-precision representations more or less
   useful is itself unknown and worth measuring before committing to a bit width.

## 4. THE GATE, WHICH IS NOT A PERFORMANCE GATE

**A speed gain is irrelevant unless pronunciation, speaker character and semantic quality survive.**
The comparison is against the current INT8 + BF16-prefill production baseline, using paired
audio, ASR and human listening. Waveform equality, mel-correlation and duration are
necessary and **not sufficient**: the earlier rejections passed exactly those checks. A
candidate that improves RTF and loses pronunciation is a failure, and must be recorded as a
failure rather than as a trade-off to be tuned later.

## 5. WHAT WOULD MAKE THIS WORTH STARTING

Prefill is a real cost at admission, and weight storage sets the memory floor per worker,
which is what bounds workers per box. Either would matter. Neither matters enough to spend
quality on, which is why this sits behind the C12-WIN ladder rather than inside it.
