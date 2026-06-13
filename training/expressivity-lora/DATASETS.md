# Emotional-speech datasets for `.expr` training (per language)

A survey of public emotional-speech corpora to train a `<lang>.expr` pack. What we need (see the
[README](README.md)): **multiple emotions × multiple speakers**, audio resampleable to **24 kHz
mono**, and ideally **known transcripts** (acted corpora with fixed sentences are easiest — our
`prepare_manifest.py` needs `text` per clip; spontaneous corpora need ASR first).

> ⚖️ **License matters for shipping.** Many SER corpora are **research / non-commercial** (CC-BY-NC-SA).
> Those are fine for personal/experimental `.expr` files but **cannot be redistributed/shipped**
> as a derivative. Prefer CC-BY / permissive if you want to release the pack. Always verify the
> dataset's own license before training a pack you intend to publish.

## Ready-to-use hub: `confit` (same parquet loader as our EMOVO example)

The [`confit`](https://huggingface.co/confit) HF org hosts parquet versions of several corpora with
a uniform schema (audio + emotion label), so they drop into the same pipeline as EMOVO:

| dataset | lang | emotions | clips | sr | note |
|---|---|---|---|---|---|
| `confit/emovo-parquet` | **Italian** | 7 | 588 | 48k | our worked example |
| `confit/emodb-parquet` | **German** | 7 | 535 | 48k→16k | **DE pack is immediately doable** |
| `confit/ravdess-parquet` | English | 8 | 2880 | 48k | acted, 24 speakers |
| `confit/crema-d-parquet` | English | 6 | 7442 | 16k | 91 speakers (big) |
| `confit/iemocap-parquet` | English | 4 | 5531 | 16k | dyadic |

(confit clips carry emotion labels; for fixed-sentence corpora the transcripts are known and can be
mapped by clip id, like our EMOVO example.)

## Per target language

### 🇩🇪 German — **EmoDB (Berlin)** ✅ easiest
7 emotions (anger, boredom, anxiety, happiness, sadness, disgust, neutral), 10 actors (5M/5F),
535 utterances, 48k→16k, **CC-BY** (relatively permissive). On `confit` as parquet → reuse the
EMOVO flow directly. Small (535) — expect r32/r64 to work; more data would help.
Sources: [Emo-DB](http://emodb.bilderbar.info/start.html), [confit/emodb-parquet](https://huggingface.co/datasets/confit/emodb-parquet).

### 🇪🇸 Spanish — **EmoMatchSpanishDB** (main) + MESD (supplement)
- **EmoMatchSpanishDB**: 6 Ekman emotions + neutral, **50 speakers** (31M/19F), ~2,050 elicited
  clips — good speaker diversity for voice-agnostic training. Verify license (academic).
- **MESD** (Mexican Spanish): 6 emotions, 864 **single-word** utterances, adults + children —
  word-level limits prosody learning; use as a supplement, not the base.
- (In-the-wild: EMOVOME spontaneous voice messages — needs ASR transcripts.)
Sources: [EmoMatchSpanishDB (UPM)](https://oa.upm.es/80921/), [MESD](https://data.mendeley.com/).

### 🇫🇷 French — **CaFE** (quality) + **Oréau** (standard accent)
- **CaFE** (Canadian French): 6 emotions + neutral × 2 intensities, **12 actors** (6M/6F),
  192 kHz/24-bit (excellent), **CC-BY-NC-SA** (non-commercial). Accent = Canadian.
- **Oréau** (standard/European French): 7 emotions, **32 speakers**, ~79 utterances (few per
  emotion), non-commercial. Standard accent but small.
- Best: combine — CaFE for clean acted range, Oréau for European accent. Both NC → personal use.
Sources: [CaFE](https://dl.acm.org/doi/10.1145/3204949.3208121), Oréau (Zenodo/SER-datasets list).

### 🇵🇹 Portuguese — **VERBO** (Brazilian; main)
- **VERBO**: acted emotional speech, Brazilian Portuguese — the most usable acted SER corpus for PT.
- **CORAA SER** (BR, spontaneous): only 3 classes (neutral / non-neutral M/F) → too coarse for our
  per-emotion instructs; not ideal.
- **European Portuguese**: only small acted corpora in the literature; scarce — BR is the practical base.
Sources: [VERBO](https://thescipub.com/abstract/jcssp.2018.1420.1430), [CORAA SER](https://github.com/rmarcacini/ser-coraa-pt-br).

## Multilingual / aggregators (useful for scale or extra languages)
- **ESD** — Emotional Speech Database: EN + ZH, 10+10 speakers, 5 emotions (clean, acted).
- **CAMEO** — Collection of Multilingual Emotional Speech Corpora ([arXiv 2505.11051](https://arxiv.org/html/2505.11051)) — a curated multilingual set.
- **EmoBox** / [SuperKogito/SER-datasets](https://github.com/SuperKogito/SER-datasets) — big indexes of SER corpora by language + license.
- **nEMO** — Polish (9 actors, 6 emotions) if you want a Polish pack.

## Transcripts — published, not invented

Acted corpora use a **fixed sentence set** that's published in the dataset's paper/docs — you map
each clip's code to its sentence, you don't transcribe by hand:

- **EmoDB (DE)** — 10 sentences, codes `a01..b10` (source: audeering audformat reference + Burkhardt
  et al. 2005). **Already built into `prepare_manifest.py` (`--emodb`).** Use the *original* EmoDB
  download (filenames like `03a01Fa.wav` encode text+emotion); the `confit` parquet drops the codes/text.
- **EMOVO (IT)** — 14 sentences (the worked example, also built in).
- **CaFE (FR)** — 6 sentences (published with phonemic transcriptions in the CaFE paper).
- **RAVDESS (EN)** — 2 fixed statements ("Kids are talking by the door" / "Dogs are sitting by the door").
- **MESD (ES)** — single-word list (published).
- **EmoMatchSpanishDB (ES) / VERBO (PT)** — check the paper/repo for the prompt set; elicited/acted
  corpora ship their sentence list. Spontaneous corpora (EMOVOME, CORAA) have **no fixed script** → run ASR.

So for the acted EU corpora the transcripts come straight from the dataset's own documentation; only
the spontaneous ones need ASR.

## Practical recommendation (priority order)
1. **German** — EmoDB via `confit` (ready, CC-BY) — fastest second language after Italian.
2. **Spanish** — EmoMatchSpanishDB (50 speakers) — best speaker diversity.
3. **French** — CaFE + Oréau (note Canadian vs European accent; NC license = personal use).
4. **Portuguese** — VERBO (Brazilian).

For each: aim for ≥ a few hundred clips across ≥ several speakers and the full emotion set; map each
emotion to a vivid **English** instruct (see `prepare_manifest.py`'s `EMOTION_INSTRUCT`); then run the
4-step pipeline. More & varied data → richer expressivity and better language-prosody/timbre.

---

# Paralinguistics (nonverbal vocalizations) — sighs / laughs / breaths

A SEPARATE track from the emotion packs above: train a LoRA that emits **non-verbal vocalizations
INLINE inside running speech** (speech → `[sigh]` → speech). This needs corpora where the event is
**transcribed inline** within the sentence — NOT isolated SFX clips (VocalSound / AudioSet "Laughter"
classes train a *classifier*, useless for TTS). Ingestion: **`prepare_nonverbal.py`** (sibling of
`prepare_manifest.py`; handles HF in-memory audio + emoji→`[marker]` inline mapping).

## Survey verdict (2026-06-12, deep-research, 18 sources, adversarially verified)

**No dataset is permissive + inline-tagged + multilingual + clean all at once** — every candidate
fails ≥1 axis, and **there is ZERO cross-lingual inline-paralinguistic data**: only **English** and
**Mandarin** inline corpora exist (no IT/DE/ES/FR tagged inline).

| dataset | inline tags? | license | lang | audio | verdict |
|---|---|---|---|---|---|
| **NonverbalTTS** (`deepvk/NonverbalTTS`) | ✅ **best** — 10 events mid-sentence, as **EMOJI** (🤣🌬😤…) | ⚠️ **CC BY-NC-SA** annot. + audio inherits VoxCeleb/Expresso (HF `apache-2.0` tag is **contradicted** by the README/paper — treat as **research-only**) | EN | VoxCeleb=YouTube (not studio) + Expresso | **#1 for the THEORY proof** (not shippable) |
| **NVSpeech** ([arXiv 2508.04195](https://arxiv.org/html/2508.04195v1)) | ✅ true inline tokens `"…funny [Laughter]"` | ⚠️ unverified; manual subset Private, auto-labeled Public | **ZH only** | unverified | largest (573h); ZH-only |
| **Expresso** (Meta) | ❌ non_verbal/laughing are **STYLE clips**, not inline | ❌ **CC BY-NC** | EN | ✅ 48kHz studio | clean but NC + not inline |
| **LibriTTS-R** ([OpenSLR 141](https://www.openslr.org/141/)) | ❌ **zero** paralinguistic tags | ✅ **CC BY 4.0 (shippable!)** | EN | ✅ 24kHz clean | the only commercial base → **auto-tag it ourselves** |
| EARS / RAVDESS / IEMOCAP / Switchboard-Fisher | ❌ clips or 2 fixed sents | ❌ NC / LDC research-only | EN | EARS 48k; SWB 8kHz tel. | **eliminated** |

**Net:** everything with inline tags is research-only; the only commercial-clean base (LibriTTS-R)
has no tags. → **plan: validate the theory on NonverbalTTS first; if it works, build a SHIPPABLE
corpus by auto-tagging LibriTTS-R (breath/laugh detection) or revisit NVSpeech/newer sets.**

**Cross-lingual hypothesis (the reason this track is worth it):** paralinguistics are ~**language-
independent** (a sigh is a sigh — the *opposite* of emotion, which we measured to be language-specific).
So an **EN-trained** paralinguistic LoRA may **transfer onto Italian/other-language speech** → we may
not need per-language tagged data. **This is the #1 thing the proof must test.**

**Follow-ups not yet audited** (could improve the cross-lingual/license picture):
**Emilia-NV** (`amphion/Emilia-NV`, "NV"=nonverbal, Emilia is large + multilingual) and Meta's
**SeamlessExpressive**. Also the richer EN candidates (LibriTTS-R as the shippable auto-tag base; any
2024-26 expressive EN set) for the "which to actually use" decision *after* the theory is validated.

## Usage — NonverbalTTS proof

```bash
# 1. Inspect the REAL emoji taxonomy first (don't trust a hand-copied table):
python3 prepare_nonverbal.py --histogram          # prints emoji freq + how each maps
# 2. Build the inline-tagged 24kHz manifest (drop low-DNSMOS clips for cleanliness):
python3 prepare_nonverbal.py --split train --min-dnsmos 3.0 --out_dir data_nv
# 3-4. Then the SAME downstream as the emotion packs:
#   upstream prepare_data.py (add audio_codes) -> train_lora.py (L16-26) -> export_expr.py
```

The emoji→marker map lives in `prepare_nonverbal.py` (`EMOJI_MARKER`). Unknown emoji are **stripped**
(raw emoji disturb the model and do nothing useful passed through). The SAME map is intended to also
power a **user-facing emoji-in-prompt** feature (known emoji in `--text` → our `[marker]`; unknown →
strip) — to be wired C-side, see PLAN.

> ⚠️ **NonverbalTTS is research/prototype ONLY** (CC BY-NC-SA + VoxCeleb/Expresso audio terms). Use it
> to prove the LoRA learns nonverbals and that EN→IT transfer works; do **not** ship a pack trained on
> it. For a shippable pack, auto-tag a permissive base (LibriTTS-R, CC BY 4.0) instead.

## Next-language survey — RU / PT / JA / KO (verified 2026-06-13)

Survey + adversarial license check for the four remaining languages (IT/DE/ES/FR already locked).
Goal: one shippable emotional corpus per language. **License is the gating axis** — we publish derived
LoRA weights, so research-only / non-redistributable corpora are LOCAL-PROOF-ONLY (train + keep private,
don't ship the pack).

### 🇷🇺 Russian — ✅ RECOMMENDED: **RESD** (cleanest license, ready now)
- **RESD** — HF `Aniemore/resd` (+ `resd_annotated`). ~**3.5h**, **1,396 utt** (1,116 train + 280 test),
  **7 emotions** (anger, disgust, enthusiasm, fear, happiness, neutral, sadness), actor-performed.
  **License MIT** (research + commercial + **redistribution of derived weights OK**). Direct HF parquet
  (~486MB), **ungated**. → **the cleanest pick of the whole survey; ship-safe.**
- **Dusha** (alt, NOT recommended) — ~350h / 300k+ recs but only **4 emotions**, crowdsourced (Yandex
  Toloka, background noise) + podcast. **License PDF-only** (no commercial/redistribution statement) and
  the **podcast audio is non-redistributable**. Too big + dirty + license-murky for our tiny-LoRA need.

### 🇧🇷 Portuguese — ⚠️ VERBO (only real option, license to confirm before SHIP)
- **VERBO** — Brazilian PT, **7 emotions** (happy/disgust/fear/neutral/anger/surprise/sad), **12 speakers**
  (6F/6M), 14 phrases, **1,167 recs**. The *only* verified discrete-emotion BR-PT corpus. Site/GitHub:
  `jrtorresneto/VERBO-emotional-speech-dataset`. **CAVEAT: marked "restricted, research-purposes only"**
  → fine for a LOCAL proof, but **confirm license / get author permission before publishing the pack**.
- **CORAA-SER** ❌ ruled out — ~50min, only 3 coarse classes (neutral / non-neutral F / non-neutral M),
  spontaneous. **TTS-Portuguese** ❌ — single-speaker neutral read TTS, no emotion (CC-BY-4.0).
- PT is Romance → should **transfer easily like ES/FR** (high payoff for little data); license is the
  only blocker to a *shippable* pack.

### 🇯🇵 Japanese — ✅ RECOMMENDED: **JVNV** (studio, share-alike OK)
- **JVNV** — HF `asahi417/jvnv-emotional-speech-corpus`. **6 emotions**, **4 pro speakers** (2F/2M),
  **3.94h / 1,615 utt**, **studio 48kHz/24-bit anechoic** (downsamples cleanly 48→24k). **License
  CC-BY-SA-4.0** → commercial + redistribution OK **under share-alike** (derived pack must carry SA).
  → ship-safe with the SA obligation.
- **STUDIES** (alt) — ~8h, 3 actors, per-line emotion labels, 48kHz studio, higher quality BUT license
  is **research-only + "please refrain from redistribution"** → **blocks public pack release.** Local-only.

### 🇰🇷 Korean — ❌ UNRESOLVED (needs a dedicated follow-up)
- **No Korean corpus verified this round.** KESS + AI-Hub emotional sets (KESDy18/19 etc.) named but
  unconfirmed; **AI-Hub terms are typically restrictive** (signed agreement / Korea-residency /
  non-redistribution) and likely **block public LoRA release**. → needs its own survey to find ONE
  downloadable + redistribution-permissive KO emotional set, OR we defer KO.

### Priority order (license-clean + downloadable + base-need)
1. **🇷🇺 RU — RESD** — cleanest license (MIT), ungated, downloadable NOW, and RU is non-native (needs the
   data most). **Best first target.**
2. **🇧🇷 PT — VERBO** — Romance, transfers easily = high payoff; train the proof now, **gate the public
   ship on confirming VERBO's license.**
3. **🇯🇵 JA — JVNV** — ship-safe (CC-BY-SA), but JA may be **near-native** in Qwen3-TTS → likely only an
   *emotion* LoRA needed, not language-rendering. **Free check first: listen to base JA (ryan/vivian) BEFORE
   training** to confirm we only need emotion.
4. **🇰🇷 KO — blocked on data** — run the KO follow-up survey (or the free base-quality ear-test) before
   committing effort.

> **UNVERIFIED hypothesis (carry forward):** that JA (and maybe KO) prosody is already near-native in
> Qwen3-TTS (emotion-LoRA-only) while RU (Slavic, non-native) needs a real language-teaching corpus —
> *plausible, consistent with the model's ZH+JA+EN training, but NOT verified against the model card.*
> The cheap test is the **free base ear-check** per language before spending on a dataset.
