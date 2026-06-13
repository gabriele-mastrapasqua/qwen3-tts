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
Goal: one usable emotional corpus per language.

> 🔓 **LICENSE CONSTRAINT RELAXED (user 2026-06-13).** We train only a **micro-LoRA per language and do
> NOT redistribute the datasets**, so **NonCommercial / research-only / non-CC corpora are FINE** as long
> as they're usable, well-made, correctly emotion-tagged, and (ideally) lightweight. (Caveat noted: an NC
> source could in theory limit *commercial* shipping of the derived LoRA — acceptable for these
> experimental packs.) → The real ranking axis is now **DOWNLOADABILITY** (instant vs form/email/gated),
> not the license. The "ship-safe" tags below still record which packs *could* be published, but no longer
> gate which datasets we may USE.

**Instant-download + well-tagged + light:** only **RESD** (RU) and **JVNV** (JA). PT (VERBO) needs an
email request; KO (ETOD/KESDy18) needs a form for the full set. See per-language detail.

### 🇷🇺 Russian — ✅ RECOMMENDED: **RESD** (cleanest license, ready now)
- **RESD** — HF `Aniemore/resd` (+ `resd_annotated`). ~**3.5h**, **1,396 utt** (1,116 train + 280 test),
  **7 emotions** (anger, disgust, enthusiasm, fear, happiness, neutral, sadness), actor-performed.
  **License MIT** (research + commercial + **redistribution of derived weights OK**). Direct HF parquet
  (~486MB), **ungated**. → **the cleanest pick of the whole survey; ship-safe.**
- **Dusha** (alt, NOT recommended) — ~350h / 300k+ recs but only **4 emotions**, crowdsourced (Yandex
  Toloka, background noise) + podcast. **License PDF-only** (no commercial/redistribution statement) and
  the **podcast audio is non-redistributable**. Too big + dirty + license-murky for our tiny-LoRA need.

### 🇧🇷 Portuguese — ⚠️ VERBO (only real option, license to confirm before SHIP)
**✅ SOLVED via Romance transfer (user ear-confirmed 2026-06-13) — NO PT dataset needed.** PT is Romance
like ES/FR/IT, and the model's EN→Romance switch is strong: the **Spanish `.expr` pack applied to
Portuguese text emotes correctly** (user A/B: ES pack on PT clearly lifts emotion). → **Suggested PT pack
= the Spanish broad-band pack** (shipped as a labelled copy `presets/expr/portuguese_bb027_r32.expr`,
same weights as `spanish_bb027_ep5_r32`, lang header patched to Portuguese). Use `-l Portuguese --expr
portuguese_bb027_r32.expr --expr-weight ~0.6` + EN instruct. This sidesteps the dead VERBO download.

- **VERBO** (the only real BR-PT discrete-emotion corpus, 7 emotions / 12 speakers / 1,167 recs) is
  **effectively DEAD**: the official site's download form 404s and the author email (jrtorresneto@usp.br)
  is unresponsive (user tried). Kept here only for the record; not obtainable.
- **CORAA-SER** ❌ (only neutral/non-neutral) and **TTS-Portuguese** ❌ (single-speaker neutral) — no
  discrete emotions. **F5-TTS-pt-br** is a *model*, not a dataset. → no usable PT corpus exists; the
  Romance-transfer pack is the answer.
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

### 🇰🇷 Korean — ✅ USABLE under relaxed license: **ETOD** (well-tagged, light; NC, research)
No Korean emotional corpus is permissively-licensed (all NC/gated/unlicensed), but with the relaxed
constraint they're now USABLE. Ranked by quality + downloadability:
- **EmotionTTS Open DB (ETOD)** — [`github.com/emotiontts/emotiontts_open_db`](https://github.com/emotiontts/emotiontts_open_db)
  → `Dataset/SpeechCorpus/Emotional/{plain-to-emotional,emotional-to-emotional}`. **15 speakers** (plain
  2F/3M + emotional 5F/5M), **4 emotions** 일반/기쁨/화남/슬픔 = **neutral/happy/angry/sad**, **100
  sentences/emotion/speaker** (full = ~6,000 clips). **Cleanly tagged** (emotion = filename range, e.g.
  `emaNNNNN`: 00001-100 neutral, 101-200 happy, 201-300 angry, 301-400 sad) **+ transcripts** (`script/`
  + `transcript/`, UTF-8). **PCM WAV 16-bit, 22.05kHz mono** (light; resample up to 24k). License
  CC-BY-NC-SA-4.0 (research, no-commercial — OK now). **DOWNLOAD REALITY: the GitHub repo is only the
  5% public sample** (~5 clips/emotion/speaker ≈ **~300 clips total, instant**); the **full 6,000-clip
  set needs a Google-form request** (free, research) linked in the repo's Emotional/README.md. → **best
  KO pick**: grab the 5% for a quick proof, or submit the form for the full set.
- **KESDy18 / KESDy19 (ETRI)** — [`nanum.etri.re.kr/share/kjnoh2/KESDy18`](https://nanum.etri.re.kr/share/kjnoh2/KESDy18?lang=En_us)
  — 30 voice actors (15M/15F), 4 emotions (neutral/happy/sad/anger), 20 sentences/emotion, Shure headset
  (clean studio). Full set behind a **signed License Agreement form** (research-only). Good fallback/2nd.
- **AI-Hub 감성/감정 음성** — [`aihub.or.kr`](https://www.aihub.or.kr) — large (~50k wav, 7 emotions) but
  **account gating typically needs a Korean ID/phone** → hardest to obtain. Skip unless you have an account.
- **Kratos-AI/korean-voice-emotion-dataset** (HF) — 4 emotions, BUT listing/download returns auth-required
  (effectively gated) + no license declared. Skip.

### 🇯🇵 Japanese — JVNV best; STUDIES now also usable (relaxed)
- **JVNV** (above) — instant HF download, 6 emotions, studio, light (3.94h). **Still the JA pick.**
- **STUDIES** — [`sython.org/Corpus/STUDIES`](https://sython.org/Corpus/STUDIES/) — ~8h, 3 actors, per-line
  emotion labels, 48kHz studio. Research-only/no-redistribution (now OK to USE), but heavier than JVNV.
- **JTES** (Twitter scripts, 4 emotions, non-pro) / **OGVC** (NII consortium, application/fee) — both
  application-gated; not worth it given JVNV is instant.

### Priority order (relaxed license → rank by downloadability + base-need)
1. **🇷🇺 RU — RESD** — instant HF download, MIT, RU non-native (needs data most). **In training now.**
2. **🇯🇵 JA — JVNV** — instant HF download, 6 emotions studio, light. But JA may be **near-native** in
   Qwen3-TTS → **free ear-check of base JA (ryan/vivian) FIRST**; likely only an *emotion* LoRA needed.
   Second target precisely because it's the other instant-download set.
3. **🇧🇷 PT — ✅ DONE, no dataset** — use the **Spanish pack on PT text** (Romance transfer, ear-confirmed);
   shipped as `portuguese_bb027_r32.expr`. VERBO is dead; no PT corpus needed.
4. **🇰🇷 KO — ETOD** — well-tagged/light/studio (4 emotions + transcripts), usable now. **5% sample is
   instant** (quick proof); full 6,000-clip set via the repo's Google-form. KESDy18 the studio fallback.

> **UNVERIFIED hypothesis (carry forward):** that JA (and maybe KO) prosody is already near-native in
> Qwen3-TTS (emotion-LoRA-only) while RU (Slavic, non-native) needs a real language-teaching corpus —
> *plausible, consistent with the model's ZH+JA+EN training, but NOT verified against the model card.*
> The cheap test is the **free base ear-check** per language before spending on a dataset.

### Trained packs + recommended `--expr-weight` (ear-tuned 2026-06-13)
| Lang | Pack | Source | Loss | Recommended weight (ear) |
|---|---|---|---|---|
| 🇷🇺 RU | `russian_bb027_r32.expr` | RESD (MIT) | 4.64 | **0.2** (full rushes the pace; 0.2 ≈ natural, base-like duration) |
| 🇰🇷 KO | `korean_bb027_r32.expr` | ETOD 5% (NC) | 4.30 | **~0.6–1.0** (duration stable at full; sad@0.6+EN-instruct = TOP) |
| 🇧🇷 PT | `portuguese_bb027_r32.expr` | = Spanish pack (transfer) | — | **~0.6** |
| 🇯🇵 JA | `japanese_bb027_r32.expr` | JVNV (CC-BY-SA) | _training_ | _TBD after A/B_ |
All broad-band L00-27 r32, 1.7B, ~90MB factored. Apply: `-l <Lang> --expr <pack> --expr-weight <w>` +
EN instruct (`-I`) at `-T 1.1` for emotion. RU's low weight = the LoRA over-forces pace on a non-native
language; KO/PT don't (KO native-ish, PT rides the Spanish pack).
