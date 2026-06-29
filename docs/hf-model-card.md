---
license: cc-by-4.0
language:
  - en
  - it
  - de
  - fr
  - es
  - pt
  - zh
  - ja
  - ko
  - ru
tags:
  - text-to-speech
  - qwen3-tts
  - emotion
  - paralinguistics
  - expressivity
---

# qwen3-tts expressivity assets (emotion + paralinguistics)

Composable expressivity add-ons for **[qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)**
(the pure-C inference engine for Qwen3-TTS). These are **plugins layered on top of normal synthesis** —
they do not replace the base model. All for the **1.7B** model.

## What's here — organized by purpose

| folder | what it does | type | size |
|---|---|---|---|
| `expr/` | **emotion fine-tunes** (per-language CSP weight-deltas) — auto-loaded by `--emotion` | `.expr` weight-delta | 30–203 MB each |
| `steer/emotion/` | **emotion steering vectors** — auto-loaded by `--emotion` | `.qlsteer` activation dir | ~232 KB each |
| `steer/paraling/` | **paralinguistic vectors** (laugh / sigh) — auto-loaded by `[laugh]`/`[sigh]` tags in the text | `.qlsteer` | ~232 KB each |

> You normally don't touch these files directly — `--emotion <name>` and inline `[tags]` load the right ones
> automatically (see **How to activate**). The raw `--expr`/`--ml-steer` flags are an advanced manual override.

> The tiny `steer/` vectors also ship inside the GitHub repo (`presets/steer/`). The big `expr/`
> files live here on HF because they are too large for git.

### `expr/` — emotion fine-tunes
- `italian_csp_topk6.expr` — **Italian emotion default** (cleanest on presets + clones).
- `german_csp_k6.expr`, `french_csp_k6.expr` — native German / French emotion.
- `italian_csp_topk4.expr`, `italian_l1626_dense.expr` (+ `_r32`/`_r64`, `multi`/`multitag`) — earlier Italian variants (A/B / research).

### `steer/emotion/` — `ryan_{ang,sad,joy,fear,disgust,surprise}`, `galatea_{ang,sad}_ft`, `vivian_{ang,sad}_ft`
### `steer/paraling/` — `laugh_vs_cry`, `sigh_vs_laugh` (+ `.qamp` source captures to rebuild)

## How to download
```bash
# from the qwen3-tts repo:
bash download_assets.sh            # fetches expr/ into presets/expr/ (sha256-verified)
# or grab a single file:
curl -L -o presets/expr/italian_csp_topk6.expr \
  https://huggingface.co/gabrione/qwen3-tts-italian-expr/resolve/main/expr/italian_csp_topk6.expr
```
Disk: the full `expr/` set ≈ 1.4 GB; Italian-only emotion needs just `italian_csp_topk6.expr` (203 MB).

## How to activate — it's automatic

You don't wire these files by hand. The engine composes the right stack for you.

**Emotion → one flag, `--emotion`.** Pick an emotion and the engine applies the ear-validated recipe:
**a preset voice → pure STEER** (the steering vector `ryan_<emo>` @ w12 — clean, every language); **a cloned
voice → COMBINE** (the language `.expr` + steer). Use the **native preset per language** (JA `ono_anna`, KO
`sohee`, ZH `vivian`, EN/Romance `ryan`). A vivid **English** `--instruct` is an optional override.

```bash
# emotion in ONE flag — engine picks expr + steer automatically
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 --emotion sad \
  --instruct "Speak softly, with quiet sadness." \
  --text "Allora, lascia che ti spieghi come stanno le cose." -o sad.wav
```
`--emotion` accepts: `sad · joy · anger · fear · disgust · surprise` (synonyms like `happy`/`angry` work too).

**Works in every Qwen3-TTS language**, not just IT/DE/FR. The steering vector carries the emotion
(it's a language-agnostic activation direction); the `.expr` renders/stabilizes the language. **DE / FR / IT
have a native `.expr`**; for **ES / PT / RU / ZH / JA / KO** the engine uses the Italian pack as a universal
cross-language renderer (ear-validated on IT/ES/RU/ZH/JA/KO). Just set `-l <Language>` and go.

**Paralinguistics → inline `[tags]` in the text.** No flags: write `[laugh]` or `[sigh]` in `--text` and
the engine performs the event automatically (it picks the onomatopoeia anchor + the right vector for you).

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
  --text "Che giornata... [sigh] non ce la faccio più. [laugh]" -o para.wav
```

> **Advanced / manual override.** Power users can still wire the raw pieces — `--expr <file> --expr-weight`,
> `--ml-steer <file> --ml-range 21-25 --ml-weight <w>` — and a manual flag always overrides the auto-router.
> Full recipes, per-voice weights and the tuning notes: **[docs/expressivity-assets.md](https://github.com/gabriele-mastrapasqua/qwen3-tts/blob/main/docs/expressivity-assets.md)**.

## License / attribution
**CC-BY 4.0.** The Italian emotion fine-tune uses the **Emozionalmente** dataset — please cite:
F. Catania, J. W. Wilke, F. Garzotto (Politecnico di Milano), *IEEE TASLP* 33:1142-1155, 2025,
doi:10.1109/TASLPRO.2025.3540662.
