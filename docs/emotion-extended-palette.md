# Beyond the 6 base emotions — extended palette analysis (2026-07-02)

*Analysis for extending `--emotion` past {sad, joy, anger, fear, disgust, surprise}. THE recipe
(`docs/emotion-THE-recipe.md`) stays untouched — this is about what to ADD on top of it. TODOs in*

---

## 1. Our method is now published literature (validation)

- **EmoSteer-TTS** ([arXiv 2508.03543](https://arxiv.org/html/2508.03543v1)): training-free **activation
  steering** in pretrained TTS (F5-TTS, CosyVoice2) — emotion conversion, **interpolation**, and
  **erasure** by adding weighted difference-of-activation vectors. Literally our recipe, independently
  published; also shows non-emotion directions (delivery styles) are extractable the same way.
- **CoCoEmo** ([arXiv 2602.03420](https://arxiv.org/html/2602.03420v1)): **composable mixed emotions** via
  linear composition of per-emotion steering vectors, no retraining. Direct precedent for vector-sum mixes.
- Adjacent: EmoShift (learned lightweight steer vectors), EmoDiff (intensity via soft-label α),
  **EmoSphere-TTS/++** (Valence-Arousal-Dominance sphere: angle = emotion, **radius = intensity**) — the
  cleanest continuous parameterization if we ever expose a 2D control.
- MiniMax Speech-02's production implementation = **one LoRA per emotion, dynamically loaded**
  ([arXiv 2505.07916](https://arxiv.org/pdf/2505.07916)) — architecturally our per-emotion `.expr`/steer.

Takeaway: weighted steer (our w10/w12) = state of the art; interpolation and composition are documented,
low-risk extensions of assets we already have.

## 2. What the pros expose (menus to steal from)

- **Azure**: classic styles (~30: cheerful, empathetic, whispering, shouting, terrified, unfriendly,
  newscast, customerservice…) + **styledegree 0.01–2.0** + role. New DragonHD voices: **61 styles usable
  as plain-text inline markers** (`[whispering] … [ecstatic] …`) mixing true emotions, delivery
  (whispering/shouting/quiet/**fast/slow**) and cognitive states (hesitant, skeptical, reflective…), plus
  6 paralinguistic tags (laughter, coughing, throat_clearing, breathing, sighing, yawning).
- **ElevenLabs v3**: **open-vocabulary** bracket tags ([excited], [sarcastic], [deadpan], [WHISPER],
  [pirate voice]…) — the model interprets free text. Opposite pole from Azure's fixed enum.
- **Hume** (Octave/EVI): **53-emotion taxonomy** — notable entries: Surprise split **positive/negative**,
  and heavy *cognitive* coverage (Doubt, Realization, Contemplation, Interest, Determination, Nostalgia,
  Tiredness, Triumph, Awkwardness…).
- **Fish Audio S2**: 24 basic + 25 advanced emotions (nostalgic, jealous, contemptuous, hysterical,
  resigned, guilty…) + 5 tone markers ([shouting], [whispering], [in a hurry tone], [screaming],
  [soft tone]) + 10 audio effects.
- **Step-Audio-EditX**: 14 emotions + **31 styles** (whisper, murmur, roar, shout, recite, child, older,
  news, story, advertising…).
- **OpenAI gpt-4o-mini-tts**: free-form `instructions` — same philosophy as our free-form-strength
  `--instruct` finding.

**Pattern across all of them — three distinct axes** (our flat `--emotion` currently has one):
1. **Affect** (sad/joy/anger/…) — we have 6.
2. **Delivery** (whisper, shout, quiet, murmur, hurried/fast, slow, tired, panting) — we have none as
   first-class; whisper + shout are the two most universally shipped controls in the industry.
3. **Register/persona** (newscast, storytelling, customer-service, child, elderly) — 1.7B `--instruct`
   already covers much of this free-form; document rather than build.

## 3. The free extension: Plutchik dyads from our existing 6 vectors

Plutchik's dyads are literally vector sums — with CoCoEmo as precedent, these cost **zero new training**,
just `--ml-steer` composition of two existing `ryan_<emo>` vectors (tune relative weights by ear,
start 50/50 at total w12):

| derived emotion | mix | feasible today? |
|---|---|---|
| **contempt** | anger + disgust | ✅ both vectors exist |
| **awe** | fear + surprise | ✅ |
| **disapproval** (delusione) | surprise + sadness | ✅ |
| **remorse** | sadness + disgust | ✅ |
| bittersweet/nostalgia | joy + sadness | ✅ (classic mixed-emotion test case) |
| outrage | anger + surprise | ✅ |
| despair | fear + sadness | ✅ |
| optimism | anticipation + joy | ❌ needs an *anticipation* primary |
| love | joy + trust | ❌ needs *trust* |
| aggressiveness | anger + anticipation | ❌ |

Also free: **intensity semantics** — our steer weight already is Azure's styledegree; documenting
`--emotion-weight` on a 0–2-like scale (w6=mild, w12=default, w16=strong if it survives the ear) aligns
with user expectations. And **erasure** (EmoSteer): subtracting a vector to *de-emotionalize* an
over-expressive clone is a documented trick worth one experiment.

## 4. New primaries / delivery directions (same extraction recipe as para vectors)

The para steering work already proved the pipeline: contrast pairs → activation diff → L21-25 RAW vector.
Candidates ranked by industry value and expected separability (remember the para lesson: contrast the
CONFUSABLE OPPOSITE, not neutral):

1. **whisper** (delivery) — contrast vs *shout*, not neutral. Highest industry value; EmoSteer explicitly
   found whisper-like directions. Also the safest: strongly acoustic, minimal semantic entanglement.
2. **shout/projected** — the same pair, opposite sign.
3. **tired/sleepy** — contrast vs excited; pairs naturally with `[yawn]` (composition demo).
4. **tender/warm** ("affectionate") — contrast vs cold/detached; Hume/Fish both ship it.
5. **sarcastic/deadpan** — contrast vs sincere/enthusiastic; hard (prosody-semantic), high wow-factor.
6. **calm/soothing** — contrast vs anxious; useful as the "erasure-adjacent" direction.
7. **anticipation** + **trust** primaries — only if we want to unlock the remaining Plutchik dyads
   (optimism, love); lower priority, fuzzier acoustic identity.

Data sources for contrast clips: our own generations via `--instruct` (1.7B interprets "whispering"
etc. weakly but enough to build contrast sets), Emozionalmente/EMOVO for affect, or Step/Fish-style
style prompts re-rendered. Same capture tooling as the emotion vectors.

## 5. Interaction with THE recipe (do-not-break rules)

- Mixes ride the SAME slot as a single emotion steer (still L21-25, still w12 total, still preset→STEER /
  clone→COMBINE). No new mechanism; `--emotion contempt` would resolve to a 2-vector sum in
  `resolve_emotion_recipe`/`EMOTION_CELLS` terms.
- Delivery vectors are a NEW composable slot (affect + delivery simultaneously, e.g. sad+whisper —
  CoCoEmo shows 2-vector composition works; must ear-test weight budget: two vectors at w12 each likely
  over-steers → start w8+w8).
- Para `[tag]`s already compose via `compose_from_text`; delivery adds a third layer — test the triples
  (emotion + delivery + [tag]) only after pairs are validated.
- Everything gated by the usual: ear-first, seed-pinned comparisons, per-experiment dated folder.

## 6. Proposed validation order

1. **contempt + awe + nostalgia** (pure vector sums, one afternoon, ear-judged) — if ≥2 of 3 read
   correctly cross-language on ryan, the dyad shelf is real.
2. **whisper vector** (new extraction, contrast vs shout) — the single highest-value addition.
3. disapproval/remorse/outrage/despair sweep + weight tuning.
4. shout, tired, tender.
5. Decide exposure: named presets (`--emotion contempt`) vs raw mixing flag
   (`--emotion-mix anger:0.5,disgust:0.5`) — recommend BOTH: names for the curated winners, the flag for
   power users (mirrors the Azure-enum vs ElevenLabs-open-vocab split).
