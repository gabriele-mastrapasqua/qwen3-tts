# Turin AWS qualification checklist — C12-WIN specs 10 / 11A / 12

**Task.** Define the bounded cloud campaign that gives specs 12, 11A and 10 a verdict, with
the OSS Base model path as the PRIMARY qualification path. **Question.** In what order, on
which model paths, with which gates, so that no cloud hour is spent on a duplicate campaign
and no verdict is confounded by two changes at once? **Known facts.** All three flags are
default off and no product profile names them; spec 12 and 11A have never executed on x86;
spec 10 is locally exact on contract A and blocked on contract C. **Unknowns.** Whether the
Base OSS SERVING path carries the qvoice conditioning into a slot correctly: the graft works
on Base on the CLI, but the clone conditioning path has never been exercised through the
batched server. Phase A0/A below is that confirmation and is NOT YET RUN.

## 1. Why the primary model path changes

The previous Turin qualification ran entirely on the CustomVoice preset-speaker path
(`-d qwen3-tts-1.7b -s ryan`). Base and CustomVoice are close in tensor geometry, so raw
kernel timing should transfer — but the SERVING INPUT PATH is materially different:

| | conditioning | how the prompt is built |
|---|---|---|
| CustomVoice | preset speaker id | speaker-ID path |
| Base OSS | voice clone / qvoice | clone conditioning path |

Similar weights do not imply identical serving behaviour. A qualification that never
exercises the clone conditioning path cannot claim the engine is qualified for it: the
prompt layout, the conditioning state carried into the slot, and the per-request setup all
differ. At least one real Base OSS qualification is required to show there is no
conditioning-path defect, no scheduler/state divergence and no path-specific quality
regression.

## 2. PHASE A0 — Base OSS + qvoice smoke (cheap confirmation, not a research question)

The grafts are portable across model paths: a qvoice generated from the 1.7B works on the
Base models too, and CustomVoice was preferred historically for the emotion levers, not
because the qvoice format is CV-only. So the expected primary invocation is:

```
bash download_model.sh --model base-large     # -> qwen3-tts-1.7b-base (public)
bash download_voices.sh                       # -> voices/galatea_graft.qvoice (CC0, sha256-verified)
./qwen_tts -d qwen3-tts-1.7b-base --load-voice voices/galatea_graft.qvoice --icl-only \
    -l Italian --text "<short IT sentence>" -T 0 --seed 42 --int8 -o a0_it.wav
```

Facts worth carrying into the run, verified in the repository 2026-09-10:

* 1.7B grafts are ~25 MB, 0.6B grafts ~16.8 MB; `voices/galatea.bin` is an 8 KB x-vector.
  `voices/HF_VOICES_README.md` describes the graft as *x-vector + TPAD + WOVR, no multi-GB
  WDELTA* — which is exactly why it is portable.
* `--icl-only` is the normal mode: it keeps the loaded weights intact and uses the ICL
  prefix. `main.c` refuses a WDELTA-carrying `.qvoice` on a Base model WITHOUT `--icl-only`
  ("this would corrupt weights"). If that error ever appears, the invocation is wrong, not
  the asset.

A0 is therefore a five-minute smoke, not an investigation: one short Italian and one short
English sentence, CLI, deterministic seed, listen once. **Its purpose is that the Base path
has never been exercised in SERVING** — only on the CLI — so Phase A must confirm the server
carries the clone conditioning into a slot correctly before any timing claim is built on it.

This document was written on a machine without the Base model present, so A0 is recorded
here as NOT YET RUN rather than assumed.

## 3. Model matrix — bounded

* **PRIMARY:** 1.7B Base OSS + Galatea qvoice (whichever candidate A0 selects).
* **SECONDARY CONTROL:** 1.7B CustomVoice + Ryan (the previously qualified path).

No duplicate long campaigns. The secondary exists only to catch a gross model-path-specific
reversal.

## 4. Phases

### PHASE A — functional smoke, both model paths
c=1 and c=2, short EN + short IT, deterministic seed, save all audio. Confirms both paths
serve correctly before any timing claim. Cheap, and it is the only place both paths run the
full server before Phase D.

### PHASE B — kernel microbench, specs 12 and 11A
Kernel timing does not depend on the conditioning path, so use whichever path adds no setup
confound; the existing CustomVoice microbench harness is acceptable here and is the reason
this phase is listed before the Base A/B.

**Spec 12 — `QWEN_SD_GLUE=1`.** First real x86 gate is `--self-test`: the V2 cases do not
execute on ARM, so this is the first time the fused context+residual path runs at all. If it
fails, STOP — no benchmark. Then the decoder microbench, `taskset -c 4-7`, B1/B3/B4,
chunk 4/8, 5 warm reps. GO: **>= 3 ms** useful decoder-unit reduction at the relevant q4
B3/B4 geometry, plus bit-identical WAV where the spec requires it. <= ~1 ms or noise is a
NO-GO for this implementation; slower is an immediate NO-GO. No "improve it a little".

**Spec 11A — `QWEN_SD_CONVT_STACK=1`.** Same machine, mask, build and provenance. GO:
**>= 2 ms** useful reduction with numerical parity within spec (the ARM self-test already
reports ~8e-8 against the per-tap reference across every block geometry over two
consecutive streaming units). If it passes, also measure whether decoder unit residency
drops and whether the CP overlap cost changes — do not infer a server benefit from isolated
ConvT time.

Do not implement Spec 11B until 11A and 12 have their evidence.

### PHASE C — server A/B for the microbench winners, PRIMARY path
Base OSS + Galatea, frozen profile, one mechanism at a time. Short bounded A/B first, then
the C12 10-minute soak only for a treatment that survives it. No soak after a failed short
A/B.

### PHASE D — secondary CV spot-check
One short run on CustomVoice + Ryan, only to confirm no gross model-path-specific
performance reversal. Not a second campaign.

### PHASE E — spec 10, separately
Section 10 of `.work/c12-win-admission-slicing-implementation.md`: the `admit_ms`
diagnostic, then the A/B, then contract C's quality bank. Spec 10 is never bundled with a
decoder treatment.

## 5. Invariants for the whole campaign

* One mechanism at a time. Never two flags in one arm.
* A failed gate is an IMPLEMENTATION NO-GO for that implementation, never a verdict on the
  mechanism. Record which of the two it is, explicitly.
* Never modify the product profile to rescue a treatment.
* Report the cold prefix-cache fallback separately from steady sliced admissions.
* Every claim carries its arm, its mask, its build and its provenance.

## 6. Known blocker, tracked elsewhere

A server-vs-CLI Italian pronunciation defect is under investigation on a separate track and
is NOT owned here. It predates these three flags and spec 10 is default OFF, so it must not
be attributed to them. It blocks a final PRODUCT QUALITY claim, but it does not block the
Phase B kernel microbenches or the Phase C diagnostic A/B, which are timing evidence. No new
runtime path is promoted until that defect is understood well enough to show the treatment
does not worsen it.
