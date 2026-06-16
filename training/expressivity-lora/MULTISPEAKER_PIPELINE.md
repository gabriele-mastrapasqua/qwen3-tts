# Multi-speaker emotion FT — reproducible pipeline + findings (2026-06-16)

Goal: make emotion **generalize** (across identities, and ideally to cloned voices) by fine-tuning the
dense expressivity band **L16-26** on emotion spoken by **many speakers × many emotions**, instead of
the EMOVO-only (6 Italian actors) set that produced a voice-specific result.

## TL;DR of what we learned (read before re-running)

1. **The mechanism works — in the dominant training language.** A dense L16-26 FT on
   ESD (10 EN speakers) + CREMA-D (91 EN actors) + EMOVO (6 IT actors) made **English** emotion
   clearly better on a preset (ryan): anger = articulate+nervous+angry, sad = slow/heavy. Ear-confirmed.
2. **Untagged language mixing CORRUPTS the minority language.** The set was **97.7% English**
   (24 942 EN vs 588 IT) and the FT builder passed **no language tag** → the shared L16-26 weights
   learned English-dominant emotion and **wrecked Italian** on ryan ("foreigner speaking poor Italian",
   dark timbre, slurring). EMOVO-only worked before precisely because it was a *single* language.
3. **Even upstream Qwen `finetuning/dataset.py` drops the language token** (reads `language`, never uses
   it) → the official recipe is implicitly **single-language FT**. Mixing languages is the deviation.
4. **The fix = condition on language** (this dir's `*_lang.py`): inject the same language codec token the
   C engine uses at inference — `[THINK, THINK_BOS, language_id, THINK_EOS, speaker, PAD, BOS]`
   (`qwen_tts.c`) — so one mixed FT can route emotion per language. Alternative: per-language adapters.
5. **The CLONE problem is SEPARATE.** A cloned `.qvoice` graft still resists emotion (its x-vector is
   out-of-distribution); no amount of multilingual emotion data fixes that. Different lever
   (better speaker embedding / disentanglement). Don't conflate it with the FT-quality work above.
6. **Temp** ≤ 1.3 (T1.5 slurs Italian). Instruct in **English** (model is EN/ZH-centric).

## Files (this dir)

ORIGINALS (pulled from the DGX as a tracked trace — do NOT edit, they are the proven recipe):
- `dgx_sft_expr.py` — dense full-rank FT, trains L16-26 (`--layers`) + text_projection, voice-agnostic.
- `dgx_dataset_expr.py` — its dataset builder (instruct-conditioned, **no** language tag).
- `dgx_sft_expr_lora.py` — the LoRA variant (low-rank, same band).
- `dgx_emovo_prep.py` — EMOVO → train_raw schema (the schema everything else mirrors).
- `prepare_data.py` — codec-encode step (Qwen tokenizer, GPU). Original upstream.
- `prepare_esd.py` — ESD (HF `duanyu027/ESD`) → schema. English speakers 0001-0010.

NEW (this epic):
- `prepare_cremad.py` — CREMA-D (HF parquet mirror) → schema. 91 actors, 12 fixed sentences.
- `concat_manifests.py` — merge + validate manifests; `--langs` stamps a per-file `language`,
  `--repeat` oversamples a minority language.
- `dgx_dataset_expr_lang.py` — **language-tagged** dataset builder (injects the language codec token,
  matching inference). Run `python3 dgx_dataset_expr_lang.py --self-test` to verify the prefix.
- `dgx_sft_expr_lang.py` — fork of `dgx_sft_expr.py` that uses the tagged builder (1-line diff: import).
- `dgx_multi_emotion.sh` — end-to-end orchestrator (download → prep → encode → concat → FT), idempotent
  via `<stage>.DONE` markers, fail-loud `need_file` gates, timestamped `multi_emotion.log`.
- `docker/Dockerfile` + `docker/build_img.sh` — the `qwen-ft:latest` image (ubuntu 24.04 + torch +
  torchaudio + qwen-tts). **Use this image for encode AND train** (the nvcr pytorch image has a broken
  torchaudio → tokenizer crashes).
- `../../tests/emo_score.py` — automatic SER scorer (audeering wav2vec2 arousal/valence) to rank
  expressivity variants without listening to every clip. CPU-ok.

## Reproduce (on the DGX, from `~/qwen-ft`)

```bash
cd ~/qwen-ft/docker && bash build_img.sh          # one-time: qwen-ft:latest
bash dgx_multi_emotion.sh                          # download ESD+CREMA-D, prep, encode, concat, FT (untagged)
# -> out_multi_l1626/checkpoint-final/model.safetensors

# LANGUAGE-TAGGED variant (the fix):
python3 concat_manifests.py --out multi_emotion_tagged/train_with_codes.jsonl \
    --langs Italian,English,English \
    emovo/train_with_codes.jsonl esd/train_with_codes.jsonl cremad/train_with_codes.jsonl
docker run --rm --gpus all --ipc=host -v $HOME/qwen-ft:/root/qwen-ft -v $HOME/qwen-ft:$HOME/qwen-ft \
    qwen-ft:latest bash -c "cd /root/qwen-ft/Qwen3-TTS/finetuning && \
    python3 -u dgx_sft_expr_lang.py --train_jsonl /root/qwen-ft/multi_emotion_tagged/train_with_codes.jsonl \
    --output_model_path /root/qwen-ft/out_multi_l1626_tagged --layers 16-26 --num_epochs 5"
```

Then (locally) extract the `.expr` and A/B:
```bash
mkdir qwen3-tts-1.7b-expr-multi && scp dgx:.../checkpoint-final/model.safetensors qwen3-tts-1.7b-expr-multi/
python3 tests/expr_extract.py qwen3-tts-1.7b qwen3-tts-1.7b-expr-multi presets/expr/<name>.expr --lang Italian
# A/B preset ryan IT vs EN, and (separately) the clone — see tests/emo_score.py
```

## Where everything lives

REPO (committed, `feat/server-batching`):
- `training/expressivity-lora/` — all scripts above + this doc.
- `tests/emo_score.py` (SER), `tests/expr_extract.py` (checkpoint → `.expr`), `tests/compare_audio.py` (mel).
- `presets/expr/italian_l1626_dense.expr` — the EMOVO-only Italian pack (tracked, validated).

LOCAL only (NOT committed — large / gitignored):
- `presets/expr/italian_multi_l1626_dense.expr` — UNtagged mixed pack (flawed: corrupts Italian). Diagnostic.
- `presets/expr/italian_multitag_l1626_dense.expr` — **tagged** mixed pack (the fix, run-1). 197 MB.
- `qwen3-tts-1.7b-expr-multi*/` — the pulled FT checkpoints (3.8 GB each).
- `samples/diag/` — the ryan A/B clips (see listening guide); `samples/multispk_ab*/` — clone A/B.

DGX (`~/qwen-ft/`):
- datasets: `esd/`, `cremad/`, `emovo/` (each has `train_with_codes.jsonl`); merged: `multi_emotion/`,
  `multi_emotion_tagged/`.
- checkpoints: `out_multi_l1626/` (untagged 5ep), `out_multi_l1626_tagged/` (tagged 5ep).
- logs/markers: `multi_emotion.log` + `*.DONE`, `multi_tagged.log` + `multi_tagged.DONE`.
- images: `qwen-ft:latest` (use this), `nvcr.io/...pytorch` (broken torchaudio — avoid for our code).

## Listening guide (A/B clips, in `samples/diag/`)

The decisive test = does the **language tag** stop the mixed FT from wrecking Italian?
```
# ITALIAN  (ryan, T1.1, "Domani vado al mercato...")
ryan_neu.wav / ryan_anger.wav / ryan_sad.wav            # no expr (baseline)
ryan_multi_neu.wav / ryan_multi_anger.wav / ...         # UNTAGGED mixed expr (broken: "foreign" Italian)
ryan_tag_neu.wav  / ryan_tag_anger.wav  / ryan_tag_sad.wav   # TAGGED mixed expr (the fix)
# ENGLISH  (ryan, T1.1)
ryanEN_*.wav (no expr) / ryanEN_multi_*.wav (untagged) / ryanEN_tag_*.wav (tagged)
```
Clone A/B (`samples/multispk_ab/`): `neu_<cond>.wav`, `<emotion>_<cond>.wav` for cond ∈ {noexpr,emovo,multi}.

## Status & open items (2026-06-16)

- ✅ Pipeline built, language-imbalance bug found, language-tag fix implemented + verified (self-test).
- ✅ Tagged FT (run-1, same data) trained → `italian_multitag_l1626_dense.expr`. **Ear-verdict on the IT
  cure = PENDING** (does ryan-IT stop degrading while EN stays strong?).
- ⏭ **run-2**: if the tag cures it, add MORE Italian emotional data (oversample EMOVO via
  `concat_manifests.py --repeat`, and/or a new IT corpus) to give the Italian path strength.
- 🔵 **SEPARATE problem — clone emotion**: a cloned `.qvoice` graft still resists emotion (x-vector OOD).
  Not solved by multilingual data; needs a different lever (better speaker embedding / disentanglement).
- knobs: temp ≤ 1.3 (1.5 slurs IT); instruct in English (model is EN/ZH-centric).
