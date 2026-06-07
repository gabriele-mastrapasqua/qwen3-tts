# Speculative decoding for this TTS — analysis (idea-spanning, not yet built)

A brainstorm TODO: does speculative decoding (or a relative) make sense for Qwen3-TTS in this
engine, and if so where/how? Conclusions are hypotheses to validate, not facts.

## The autoregressive structure (what we'd accelerate)

Per audio frame (12.5 Hz) there are TWO sequential stages:
1. **Talker** — generates `code0` (the coarse audio token), one per frame, autoregressive across
   frames (KV cache). ~28–31 ms/f on 0.6B M1.
2. **Code Predictor (CP)** — generates `code1..code15` (the RVQ residual codebooks) for that frame,
   **15 sequential passes**, each conditioned on the previous codebooks. It IS the model's
   "MTP" head (`small_to_mtp_projection`), but it's an **intra-frame** multi-token predictor, not a
   next-frame speculator. On 0.6B the CP is ~74–90% of the frame (the real bottleneck);
   `code0` even feeds back into the Talker (`qwen_tts.c:1300`).

So "the model has no MTP" is *almost* right: it has an intra-frame RVQ MTP (the CP), not a
speculative next-token head on the main stream.

## What speculative decoding needs
A cheap **draft** proposes K tokens; the **target** verifies them in ONE parallel forward and
accepts the longest matching prefix (with a rejection-sampling correction so the output
distribution is unchanged under temperature). Net win iff: draft is much cheaper than target,
AND acceptance is high, AND the parallel verify is cheaper than K sequential target steps.

## Ideas, spanning the space

**(A) Cross-model draft: 0.6B drafts `code0`, 1.7B verifies.** Run the 0.6B Talker to propose K
frames of `code0`, then verify all K in one **batched** 1.7B Talker forward; accept the matching
prefix. Pro: 0.6B is ~3× cheaper, weights already exist. Con: acceptance on *audio* tokens under
temperature is unknown — audio codes are noisier/higher-entropy than text, so agreement may be
low. **This is the make-or-break number and it's CHEAP to measure** (instrument-only, below).

**(B) Training-free self-speculation (Lookahead / Jacobi decoding).** Guess n-grams of future
`code0`, verify in parallel against the same Talker — no draft model, no training. Plausible for
the Talker stream; effectiveness depends on how repetitive/predictable code0 sequences are
(silences, sustained phonemes → repetitive → good; transients → poor). Medusa/Eagle are OUT
(they need trained extra heads; this is an inference-only engine).

**(C) CP residual speculation.** The CP's 15 sequential passes are the bottleneck. Draft the
*late* codebooks (c8–15 = fine texture/prosody) cheaply and verify? Risky: we measured those
residuals carry the prosody surface and drift the most — quality-sensitive. Lower expected value;
park unless (A)/(B) disappoint.

**(D) Spec-decode ⊂ batching.** The "verify K in parallel" step is exactly a **batched** forward
(weights read once, applied to K positions). So the batched-GEMM machinery we're building for
chunk-batching is the SAME primitive spec-decode needs. → Build batching first; spec-decode
becomes a natural opt-in mode on top of it, not a separate engine.

## Does it even make sense for TTS here?
- Spec-decode trades **extra FLOPs** (draft + parallel verify) for **fewer sequential steps**. It
  pays when the engine is **latency/sequential-bound** and has spare compute/bandwidth headroom.
- On **M1** single-stream is compute-bound-ish → the draft compute competes; benefit muted unless
  acceptance is high. On **bandwidth-bound x86** the parallel verify is cheap (batched read-once)
  → better fit — same conclusion as batching: x86 is where it pays.
- The honest unknown is **acceptance rate on sampled audio tokens**. Everything hinges on it.

## Cheapest first experiment (do this before any kernel work)
Instrument-only, no new math: during a 1.7B greedy synthesis, also run the 0.6B Talker on the
same context and log how often the 0.6B top-1 `code0` matches the 1.7B choice (and the top-k
overlap), greedy and at temp 0.5. That single acceptance-rate number decides if (A) is worth
building. (Reuse the teacher-forcing rails from the quant-ladder instrument — same idea: run a
second model on fixed context and compare.) If acceptance ≳60–70%, cross-model spec-decode is
worth a prototype on the batched-verify path; if it's low, audio tokens are too high-entropy and
we drop spec-decode for TTS.

## TODO
- [ ] Measure 0.6B↔1.7B `code0` acceptance rate (greedy + temp) — the decision number.
- [ ] If promising: prototype (A) on top of the batched-verify kernel (opt-in, default unchanged).
- [ ] Sketch (B) lookahead feasibility on code0 repetitiveness (cheap: histogram n-gram repeats).
