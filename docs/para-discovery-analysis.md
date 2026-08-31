# Paralinguistics discovery — mechanism + systematic (no-ear) discovery methods

*Analysis 2026-07-02. Companion to `docs/para-experiments.md` (the WIN/KO log) and `docs/para-target.md`
(the desiderata menu). This doc answers two questions: (1) WHY does the inline native-trigger method work
on Qwen3-TTS, with evidence from the tech report and the pro-model literature; (2) how to discover and map
new events/variants SYSTEMATICALLY instead of listening to every seed-sweep clip by ear.*

---

## 1. The mechanism — confirmed: emergent, not trained

The Qwen3-TTS Technical Report ([arXiv 2601.15621](https://arxiv.org/abs/2601.15621)) settles it:

- Trained on **>5M hours, 10 languages**; text goes through the **standard Qwen tokenizer** — **no
  paralinguistic special tokens are added**, no tag vocabulary is documented anywhere.
- The 12Hz codec tokenizer is explicitly designed to "fully preserve paralinguistic information and
  acoustic environmental features" — i.e. the *acoustic* side of a laugh survives into the codec tokens;
  only the *control* side (a trained tag) is missing.

So laughter/sighs/gasps in those 5M hours were transcribed by whatever the ASR pipeline emitted — and in
Chinese-heavy data that is typically **the interjection characters themselves** (哈哈哈, 唉, 嗯, 啊). Our
inline CN-onomatopoeia trigger exploits a *statistical text→acoustic association learned from noisy
transcripts*, not a control token. This explains every phenomenon in our WIN/KO table:

- **Seed dependence**: the association is statistical, so sampling decides whether the model *performs*
  the event or *reads* the characters. There is no trained token to make it deterministic.
- **嗯 multi-event (seed→meaning)**: the ASR wrote 嗯 for many different real events (mm-hmm confirmation,
  pleasure mmm, contemptuous hmpf, filled pause). The char is a superposition of all of them; the seed
  picks which mode collapses out.
- **2-class seed rule (clone vs preset)**: clone and preset prompts differ in prefix construction
  (ref-audio ICL span vs preset embedding), so the same seed lands in different sampling trajectories —
  consistent with class-level, not per-voice, behavior.
- **Articulatory ceiling (cough/sneeze)**: these are rarely transcribed as pronounceable text in any
  language, so no text→event association exists to exploit. Confirms para-target.md §"blocked".

**Precedents — same mechanism elsewhere:**

- **Bark** (suno-ai): `[laughs]`, `[sighs]`, `[gasps]`, `[clears throat]` work because bracketed stage
  directions existed in its noisy subtitle/caption training text — emergent, and notoriously
  seed-dependent, exactly like ours.
- **CosyVoice 2** ([arXiv 2412.10117](https://arxiv.org/pdf/2412.10117), Table 1): `[laughter]`,
  `[breath]` bursts + `<laughter>…</laughter>`, `<strong>…</strong>` wrappers are **plain text tokens**
  (regular tokenizer vocab, NOT special tokens), made reliable with only **~1,500 h of "instructed" data**
  mixed into training. CosyVoice 3 expands to ~13 insertion tags (`<breath>`, `<laughter>`, `<cough>`,
  `<sigh>`, `<gasp>`, `<lipsmack>`, `<hissing>`…) plus wrappers.
- **Step-Audio-EditX** ([arXiv 2511.03601](https://arxiv.org/pdf/2511.03601)): ~22 inline tags of which
  nearly half are **named Chinese interjections** — `[Confirmation-en]`=嗯, `[Question-ei]`=诶,
  `[Dissatisfaction-hnn]`=哼, `[Surprise-wa]`=哇, `[Surprise-yo]`=哟 — direct confirmation of our
  "Step labels = CN interjections" reading. Full list also includes `[sigh] [inhale] [exhale] [breath]
  [laugh] [chuckle] [giggle] [cough] [snort] [clears throat] [uhm]`.
- **Orpheus TTS**: `<laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>` — explicitly
  trained on annotated data. **SoulX-Podcast**: true special tokens `<|laughter|> <|sigh|> <|breathing|>
  <|coughing|> <|throat_clearing|>`.

**Bottom line**: our `[tag]→native-trigger+seed` map is a zero-train re-implementation of the CosyVoice
contract on top of raw statistics. CosyVoice2 proves ~1.5k h of plain-text-marker data would make it
deterministic — relevant if the para FT is ever revisited (see §5).

---

## 2. Systematic discovery — replace ear-sweeps with an offline auto-judge

The bottleneck of T3/T4 is listening to seed sweeps 1-by-1. The prior SER-judge attempt failed because a
speech-emotion classifier trained on real speech is out-of-domain on TTS audio (csp-ft-emotion.md §judge).
**Paralinguistic events are a different, easier problem**: general-purpose *audio event* classifiers are
trained on wild YouTube audio (AudioSet) and detect "is there laughter in this clip" — far more
domain-robust than 7-way emotion classification. Recommended stack (all CPU-fine on M1, Python offline
tooling — never part of the C engine):

1. **PANNs CNN14** — 81M params, 527 AudioSet classes, `pip install panns_inference`
   ([repo](https://github.com/qiuqiangkong/panns_inference), checkpoint ~300MB from
   [zenodo 3987831](https://zenodo.org/record/3987831)). Verified against the official
   [AudioSet ontology](https://github.com/audioset/ontology/blob/master/ontology.json): it has **exactly
   our target classes** — `Laughter`, `Giggle`, `Snicker`, `Belly laugh`, `Chuckle, chortle`, `Sigh`,
   `Gasp`, `Groan`, `Grunt`, `Yawn`, `Pant`, `Snort`, `Breathing`, `Cough`, `Throat clearing`, `Sneeze`,
   `Sniff`, `Crying, sobbing`, `Wail, moan`, `Whispering`, `Screaming`, `Shout`, `Hiccup`. The
   sound-event-detection variant gives framewise output → *where* in the clip the event fired (is the
   laugh at the tag position?).
2. **Whisper-AT** — Whisper encoder + audio-tagging head, outputs the **transcript AND the 527 AudioSet
   labels in one pass** at <1% extra cost (`pip install whisper-at`,
   [repo](https://github.com/YuanGongND/whisper-at)). This is the exact WIN/KO discriminator we need:
   event-class prob high **and** transcript does NOT contain the literal onomatopoeia ⇒ WIN candidate;
   literal "ha ha ha" spoken ⇒ KO-literal. Use tiny/base/small on M1.
3. **CLAP zero-shot** (`laion/clap-htsat-unfused`, ~150M, HF transformers) — score wavs against free-text
   prompts ("a person sighing sadly", "a suppressed chuckle"). Noisier on point events than CNN14; use as
   second opinion and for nuances AudioSet lacks (variant labeling for T4: "warm laugh" vs "mocking laugh").
4. **jrgillick/laughter-detection** — small ResNet laughter segmenter with onset/offset timestamps —
   purpose-built check that the laugh lands where the tag was placed.
5. Plain Whisper transcription of `[laughs]`-style annotations: unreliable/hallucination-prone — do NOT
   use as primary signal; Whisper-AT replaces it.

**Mandatory calibration step (learned from the SER-judge failure)**: before trusting the judge, run it on
the EXISTING ear-validated WIN/KO table (samples in `samples/tests/*para*`) — we have dozens of
ground-truth clips. Measure per-event precision/recall, pick per-event thresholds τ. If CNN14 agrees with
the ear on the known table, it earns referee status for *screening*; the ear stays the final judge on
promotions (same philosophy as mel-corr vs ear for emotion).

**The pipeline** (proposed `tools/para_judge.py` + `tools/para_sweep.sh`):

```
sweep:   for seed in {...}: ./qwen_tts ... --text "carrier, TRIGGER, carrier" --seed $seed -o out/s$seed.wav
judge:   CNN14 clipwise probs + Whisper-AT transcript per wav
score:   WIN-candidate  = P(target event) > τ_event  AND  transcript lacks literal onomatopoeia
         KO-literal     = transcript contains the literal
         KO-other/DRIFT = neither (or wrong-event class dominates — auto-detects serendipity WINs
                          for a DIFFERENT tag, e.g. 呜呜→[aww], per the "keep the WIN" principle)
output:  markdown table sorted by score → ear-check ONLY the top candidates
```

Throughput: hundreds of clips/hour on M1 CPU → a 5-trigger × 10-seed × 4-voice grid (200 clips) becomes
one batch run + a 10-minute ear pass on the shortlist, instead of a full listening day. This directly
unblocks: the 嗯 seed→meaning portability question, the 3rd-clone confirmation, the T4 variants grid
(~10 laughs / ~5 sighs), and even a cry hunt #2 at much wider seed range.

---

## 3. Discovery from the weights/activations (the "can we see it in the model?" question)

No published work does this for TTS paralinguistics (open territory), but three probes are cheap because
we own the forward pass:

1. **Embedding-similarity trigger mining (weights-only, no generation)**: take the text-embedding vectors
   of known-WIN triggers (哈哈哈, 唉, 嗯, 啊, 哼, 哈啊) and rank the entire tokenizer vocabulary by cosine
   similarity. Nearest neighbors are candidate NEW triggers (other interjections, other scripts —
   Japanese kana onomatopoeia ふふ/はぁ, Korean 하하, Cyrillic ха-ха…). This is a genuinely weight-based
   discovery method and costs one afternoon of Python over the mmapped embedding table.
2. **Logit-lens at event frames** (precedent: [AudioLens, arXiv 2506.05140](https://arxiv.org/pdf/2506.05140)):
   project Talker hidden states (mid-late layers) at frames where the event fires onto the codec
   unembedding — do "laugh frames" resolve to a recognizable codec-token cluster? If yes, codebook-0
   token IDs become an event fingerprint usable as an even cheaper WIN detector (no audio decode needed).
3. **Steering-vector projection as an online event detector**: we already have `laugh_vs_cry` /
   `sigh_vs_laugh` L21-25 vectors. Dot-product hidden states against them during generation → a per-frame
   "event-direction activation" trace. Cheap to add behind `--debug`; correlating the trace with WIN/KO
   seeds tells us whether the event decision is visible mid-network (it should be — act-map already showed
   L23-26 carries laugh identity).

These three feed each other: (1) proposes triggers, §2's judge screens them at scale, (2)/(3) explain the
seed classes and might eventually let us *pick* good seeds by probing instead of sampling.

---

## 4. New trigger/tag candidates from the pro vocabularies

Cross-referencing Step/CosyVoice3/Fish-S2 menus against our WIN/KO table, still-unexplored *vocal*
(non-articulatory) candidates worth a judged sweep:

| candidate tag | triggers to try | source |
|---|---|---|
| `[giggle]` (laugh variant) | 嘿嘿, 呵呵, ふふ | Step `[giggle]`, our chuckle finding |
| `[snort]` | 哼 short variants, 嗤 | Step `[snort]` |
| `[uhm]` filled pause | 呃, 诶, "uhm" | Step `[uhm]`, para-target T2 |
| `[inhale]`/`[exhale]` | 吸, 呼, "hhh" | Step — borderline articulatory, cheap to screen |
| `[wail]` (cry family retry) | 哇 long carriers, broken `啊…啊…` | AudioSet `Wail, moan` gives the judge a class for cry hunt #2 |
| `[hum]` | 嗯哼, 哼哼 (melodic) | Fish S2 |

Articulatory family (cough/sneeze/lipsmack) stays parked — the tech-report reading (§1) *confirms* the
decoder-ceiling hypothesis, don't burn sweeps there.

---

## 5. If the para FT is ever revisited (long-term note)

The FT was exhausted because bracket tags tokenize as sub-words with no gradient anchor (memory
the shipped recipe). The literature suggests the viable variant is different: **CosyVoice-style
plain-text-marker data** — transcripts that contain the *native interjection chars* aligned with real
events, no new tokens at all. **SynParaSpeech** ([arXiv 2509.14946](https://arxiv.org/html/2509.14946v2),
[repo](https://github.com/ShawnPi233/SynParaSpeech)) is an automated pipeline that mined 118.75 h of
timestamped paralinguistic events (3-ASR majority voting) — the modern recipe for building such a dataset
without manual annotation. This would train the *existing* text→event association to be seed-stable
instead of teaching new tokens — a fundamentally different bet than the failed runs, on DISJOINT layers
from the emotion plugin per the design contract. Not scheduled; recorded so we don't re-derive.

---

## 6. Actionable summary

1. Build `tools/para_judge.py` (CNN14 + Whisper-AT) and **calibrate it on the existing ear-validated
   table** before use. → then all T3/T4 sweeps become batch jobs.
2. Run the pending open questions through it: 嗯 portability (ryan/galatea × seed set), 3rd clone
   (hugo/ohenry) gasp/yawn, cry hunt #2 with `Crying, sobbing`+`Wail, moan` as the judge classes.
3. Embedding-similarity trigger mining over the vocab (weights-only, one script).
4. T4 variants grid (laugh ×10, sigh ×5) judged at scale; ear only on the shortlist.
5. Optional `--debug` steering-projection trace for mechanism insight (§3.3).
