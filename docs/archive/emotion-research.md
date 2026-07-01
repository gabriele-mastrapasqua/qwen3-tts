# How pro/SOTA TTS make emotion generalize to cloned (zero-shot) voices — research synthesis (2026-06-15)

Deep web research (103 agents, 24/25 claims adversarially verified). Question: how do production/SOTA
TTS get strong emotion control on NOVEL/cloned speakers, where our instruct+FT works on presets but is
weak on x-vector clones? Full report: task `wji0jjsku`.

## The headline (it confirms our own diagnosis)

**Pros do NOT add clone-emotion by per-emotion WEIGHT fine-tuning (exactly what we did, on preset IDs).
They DISENTANGLE emotion from speaker timbre at the REPRESENTATION level, with a separate speaker-invariant
emotion CHANNEL trained across many speakers × emotions.** Fine-tuning emotion into weights against
in-distribution preset speakers does not generalize to a novel x-vector — matches our measurement (the
sad-direction differs cross-voice; emotion lands "off" on the clone).

## Mechanisms (verified, with sources)

1. **Adversarial GRL (gradient-reversal) on the emotion embedding** — the most-validated fix. A speaker
   classifier is attached to the emotion/style embedding; a GRL reverses its gradient so the embedding
   CANNOT encode timbre → emotion control becomes speaker-invariant. **IndexTTS2** (arXiv 2506.21619):
   GRL "forces e to exclusively capture emotional and rhythmic attributes, remaining invariant to
   speaker-specific timbre." **SelfTTS** does it label-free (cosine between emotion & speaker embeddings).
   Caveat: the released IndexTTS2 checkpoint may omit the GRL head (index-tts#433).
2. **Mutual-information minimization (MINE / Donsker-Varadhan)** between separate timbre & emotion
   extractors — non-adversarial alternative (arXiv 2510.01722).
3. **Information bottleneck (iEmoTTS, TASLP 2023, arXiv 2206.14866)** — a *tunable knob*: narrower
   bottleneck → more emotion transfer, eventually erodes speaker similarity. Enables zero-shot emotion
   transfer to speakers unseen in training.
4. **Remove the utterance-level speaker vector from the LM (CosyVoice 2)** — they found the speaker vector
   "contains not only speaker identity but also language and paralanguage information, which HARMS prosody
   naturalness and CROSS-LINGUAL capability of the text-speech LM"; speaker is recovered downstream in the
   flow-matching decoder. **→ This is EXACTLY our Italian→French temp-drift pathology**: our ECAPA x-vector
   injected into the Talker LM leaks language/paralanguage and corrupts cross-lingual stability at high temp.
5. **Separate emotion CHANNEL + explicit instruction, trained jointly with cloning.** IndexTTS2 = two
   prompts (timbre prompt + a *separate* emotion prompt, may be a different speaker). CosyVoice 2 / EmoVoice
   = natural-language instruction channel. **CosyVoice 2 ablation: removing the instruction input drops
   emotion MOS-I 4.11→2.54** → emotion control does NOT emerge implicitly; it must be an explicitly trained
   channel. (Same ablation: removing instruction IMPROVES speaker similarity — there is a real
   emotion-strength ↔ timbre-fidelity TENSION to manage.)
6. **Conflicting-pair training + speaker-verification reward (FlexiVoice)** — train with the reference
   voice in emotion A but the instruction demanding emotion B, plus a CAM++ speaker-verification reward →
   forces the model to obey the instruction not copy reference prosody; holds for novel timbres (78%/76%
   instruction-emotion accuracy, only ~11-13% reference leakage).

## Data foundation
Multi-speaker × multi-emotion (× multilingual) corpora are what make emotion generalize across speakers:
**ESD** (20 speakers = 10 EN + 10 ZH, 5 emotions neutral/happy/angry/sad/surprise, 350 parallel utts/spk),
**Expresso** (Meta), **MEAD**, EmoV-DB, RAVDESS, JL-Corpus, EMOVO (our IT set, 6 speakers). Diversity of
*identities* during emotion training is the lever — not more clips of the same few actors.

## Prioritized recommendations FOR US (codec-LM + x-vector + dense L16-26 FT)

1. **CHEAPEST, do first — multi-speaker emotion data.** Re-train the emotion FT on MANY speakers × emotions
   (ESD's 10 EN speakers × 5 emotions + EMOVO for IT), not EMOVO-only (6 IT actors). The FT then learns the
   emotion transform across diverse identities → generalizes to novel x-vectors far better. Fits our existing
   `gpu_sft_expr.py` pipeline; just change the data. This directly attacks our measured "emotion direction is
   voice-specific" problem.
2. **Conflicting-pair augmentation** (FlexiVoice-lite) in data prep: pair a reference clip in emotion A with
   an instruction for emotion B, so the model learns instruct overrides the reference identity/prosody.
3. **The proper (bigger) fix — adversarial disentanglement.** Add a speaker classifier + GRL on an emotion
   representation during FT so emotion becomes speaker-invariant (IndexTTS2/SelfTTS). Needs training-code
   changes on the GPU box (new loss head), not just data — flag as the SOTA Phase-3+ track.
4. **Cross-lingual temp drift is architectural** (x-vector-in-LM leakage, CosyVoice 2). Without an arch
   change we manage it by capping temp ≤ 2.0 (already our finding).
5. **Manage the emotion↔timbre tension** (CosyVoice 2 ablation): stronger emotion conditioning costs some
   speaker similarity — tune the dose (our `--expr-weight`) per use-case.

## Sources (primary)
IndexTTS2 arXiv:2506.21619 · SelfTTS arXiv:2603.22252 · MINE-disentangle arXiv:2510.01722 · iEmoTTS
arXiv:2206.14866 · cross-speaker emotion transfer arXiv:2109.06733 · CosyVoice 2 arXiv:2412.10117 +
funaudiollm.github.io/pdf/CosyVoice_2.pdf · EmoVoice arXiv:2504.12867 · FlexiVoice (PPT/GRPO).
