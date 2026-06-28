# Expressivity assets — emotion & paralinguistics (download + how to activate)

This guide covers the **expressivity add-ons** for qwen3-tts: emotion and paralinguistic events
(laugh / sigh). They are **composable plugins** layered on top of normal synthesis — they do not
change the base model.

There are two kinds of asset:

| kind | what | size | shipped where |
|---|---|---|---|
| **steering vectors** (`.qlsteer`) | tiny activation directions (emotion + paralinguistic) | ~232 KB each (~3 MB total) | **committed in this repo** → `presets/steer/` |
| **FT weight-deltas** (`.expr`) | per-language emotion fine-tunes (CSP) | 30–203 MB each (~1.4 GB all) | **download from Hugging Face** (too big for git) |

## 1. Download the `.expr` (only needed for the `.expr`-based recipes)

The steering vectors are already in the repo. The large `.expr` emotion fine-tunes are hosted on
Hugging Face: **<https://huggingface.co/gabrione/qwen3-tts-italian-expr>**

```bash
bash download_assets.sh            # fetches the win .expr into presets/expr/ (sha256-verified)
bash download_assets.sh --verify   # check integrity of what you already have
# mirror elsewhere?  BASE_URL=<url> bash download_assets.sh
```

Disk budget: the 9 win `.expr` total ~1.4 GB. If you only want Italian emotion, you really only
need `italian_csp_topk6.expr` (203 MB). German/French add `german_csp_k6.expr` / `french_csp_k6.expr`.

### What each `.expr` is (see also `presets/expr/MANIFEST.md`)
- `italian_csp_topk6.expr` — **the Italian emotion default** (cleanest on clones + presets).
- `german_csp_k6.expr`, `french_csp_k6.expr` — native German / French emotion.
- `italian_l1626_dense.expr` (+ `_r32`/`_r64`, `multi`/`multitag`) — dense/variant Italian FTs (research/A-B).

## 2. Activate EMOTION

Emotion = a **steering vector** (the main lever) optionally combined with an **`.expr`** (for language
correction / anger / far languages). All on the 1.7B model. Per-voice steer weight matters; `w8` is the
global sweet spot, `w12` over-steers.

**A) Same / close language → STEER (max emotion):**
```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
  --text "Non posso credere a quello che è successo." \
  --ml-steer presets/steer/emotion/ryan_sad.qlsteer --ml-range 21-25 --ml-weight 8
```

**B) Far language (e.g. CJK/Cyrillic) or a voice that drifts → COMBINE (add the `.expr`):**
```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Russian -T 1.1 \
  --text "<russian text>" \
  --expr presets/expr/italian_csp_topk6.expr --expr-weight 1.0 \
  --ml-steer presets/steer/emotion/ryan_ang.qlsteer --ml-range 21-25 --ml-weight 8
```

**C) Cloned voice (graft):** same flags, replace `-s ryan` with `--load-voice voices/<clone>_graft.qvoice --icl-only`.

Available emotion vectors (in `presets/steer/emotion/`, see its MANIFEST for per-voice weights):
`ryan_{ang,sad,joy,fear,disgust,surprise}`, `galatea_{ang,sad}_ft`, `vivian_{ang,sad}_ft`.
Notes: **ryan anger** is better via `.expr` (steer goes metallic); **vivian** drifts language → prefer
`.expr` for anger; `w4` for ryan fear/surprise.

## 3. Activate PARALINGUISTICS (laugh / sigh)

Paralinguistic events are **vocal** and need TWO things together:
1. a **native-trigger onomatopoeia inline** in the text (this seeds the event):
   - laugh → put `ahah` / `hahaha` in the sentence
   - sigh  → put `haaah` in the sentence
2. the matching **paralinguistic steering vector** (this shapes/reinforces it):

```bash
# SIGH on a clone, Italian
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_graft.qvoice --icl-only -l Italian -T 1.1 \
  --expr presets/expr/italian_csp_topk6.expr --expr-weight 1.0 \
  -I "Speak in a heavy, weary, exhausted tone." \
  --text "Haaah... sono così stanca, haaah, che giornata lunghissima." \
  --ml-steer presets/steer/paraling/sigh_vs_laugh.qlsteer --ml-range 21-25 --ml-weight 8
```
Swap `sigh_vs_laugh.qlsteer` + `haaah` for `laugh_vs_cry.qlsteer` + `ahah` to laugh.

Per-voice weight: **galatea 8 · vivian 8 · ryan 6** (ryan caps at 6; `w12` over-steers everywhere).
Only the **vocal family (laugh, sigh)** works via steering; articulatory events (cough/sneeze/…) do not —
use native-trigger onomatopoeia inline (`ugh`, `ahem`, `tsk`, `haaa`) for those.

> The `[sigh]`/`[laugh]` inline-tag auto-routing (write the tag in `--text`, no flags) is being wired to
> this recipe — until then use the explicit flags above.

## 4. License / attribution
The Italian emotion FT data uses the **Emozionalmente** dataset (CC-BY 4.0) — cite F. Catania, J. W. Wilke,
F. Garzotto (PoliMi), IEEE TASLP 33:1142-1155, 2025, doi 10.1109/TASLPRO.2025.3540662 in any redistribution.
