# THE emotion recipe — the one and only (do not re-derive)

> **This is the single source of truth for emotional TTS in this engine.** When in doubt, follow THIS file —
> do NOT start from the old/historical methods (graft, x-vector, τ-vectors, `.vec` control vectors, dense FT,
> per-seed hacks…). Those were the weeks-long search; the search is over. The recipe below is ear-validated.
>
> **Two aligned copies, both authoritative:**
> 1. this doc (human-readable),
> 2. the code: `EMOTION_CELLS[]` + `resolve_emotion_recipe()` in `main.c`, exposed as the `--emotion` flag.
>
> If you change one, change the other. Provenance of the literal params: `tests/recipe_final.sh` (Italian) +
> the per-language scripts (`tests/{german,french,spanish}_ab.sh`, `tests/crosslang_emo.sh`) + the DGX checkpoints
> (`reference_dgx_ft_checkpoints`). Last ear-validation: 2026-06-24 (IT) / 2026-06-29 (per-language).

## How a user invokes it
**One flag.** `--emotion <sad|joy|anger|fear|disgust|surprise>` (1.7B CustomVoice only). The engine auto-applies
the table below (expr + steer + a default English instruct + temperature). A vivid **English** `--instruct` and an
explicit `-T` always override. On the 0.6B model `--emotion` falls back to the legacy `.vec` path.

```bash
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian --emotion sad --text "…" -o out.wav
```

## The mechanism (three modes)
- **STEER** — multi-layer activation steer `presets/steer/emotion/ryan_<emo>.qlsteer` @ `--ml-range 21-25`,
  `--ml-weight 8` (fear/surprise 4), `--ml-decay 0.985`. The ryan-captured CLEAN palette is used for **all voices**
  (it transfers cross-voice/cross-language). NO instruct on pure-STEER cells. Carries the emotion.
- **EXPR** — a per-language fine-tune `.expr` weight-delta + an English instruct, no steer. Renders/stabilizes the
  language and gives the base emotion. Used where steer goes metallic (preset anger) or the language needs the FT.
- **COMBINE** — EXPR + STEER together. The expr renders the language, the steer pushes the emotion.

## The table (voice × language × emotion)

### Italian / English (per-(voice×emotion) — `EMOTION_CELLS[]`)
| voice | anger | sad | joy | fear | disgust | surprise | T |
|---|---|---|---|---|---|---|---|
| ryan (preset) | EXPR 1.2 | STEER w8 | COMBINE 1.2/w8 | STEER w4 | STEER w8 | STEER w4 | 1.1 |
| vivian (preset) | EXPR 1.2 | STEER w8 | COMBINE 1.2/w8 | STEER w8 | STEER w4 | STEER w8 | 1.1 |
| galatea / clone | STEER w8 | STEER w8 | STEER w8 | COMBINE 1.0/w8 | COMBINE 1.0/w8 | STEER w8 | 1.1 |
expr = `italian_csp_topk6.expr`. Clone = the 25 MB graft (`--icl-only`). IT ear-validated 2026-06-24; galatea anger
= STEER `ryan_ang` (NOT galatea_ang_ft — that idea-2 dir is weaker/sad-ish).

### Other languages (per-language policy — `resolve_emotion_recipe()`), ear-validated 2026-06-29
| language | best voice | mode | expr | steer | T | verdict |
|---|---|---|---|---|---|---|
| **German** | **vivian** | EXPR | `german_csp_k6.expr` @1.2 | — | 1.1 | ✅ k6 wins (vs r32) |
| **French** | **vivian** | EXPR | `french_csp_k6.expr` @1.2 | — | 1.1 | ✅ k6 wins (vs r32) |
| **Chinese** | **vivian** (native) | STEER | — | ryan_<emo> w8 | 1.1 | ✅ good |
| **Japanese** | **galatea-graft** (clone) | COMBINE | `italian_csp_topk6` @1.0 | ryan_<emo> w8 | 1.1 | ✅ good |
| **Korean** | **galatea-graft** | COMBINE | `italian_csp_topk6` @1.0 | ryan_<emo> w8 | 1.1 | ✅ good (joy MUST be COMBINE — steer alone runs away) |
| **Russian** | **galatea-graft** | COMBINE | `italian_csp_topk6` @1.0 | ryan_<emo> w8 | 1.1 | ✅ good |
| **Spanish** | **vivian** | EXPR | `italian_csp_topk6` @1.2 | — | 1.1 | ✅ k6 wins (beat native MESD r32 AND ryan-romance @1.6) |

Notes:
- **Best voice differs per language** — the engine applies the recipe to *whatever voice you pass*, but for the
  GOLD result use the recommended voice (German/French/Chinese/**Spanish** → `-s vivian`; JA/KO/RU → the galatea
  graft clone; Italian → `-s ryan` or the clone). The router prints a hint when your voice isn't the recommended one.
- All instructs are **English** (the model follows EN/ZH, not the spoken language). Defaults baked per emotion.
- seed 42 is the reference; the per-language scripts all use it.

## Assets
- expr packs (`presets/expr/`): `italian_csp_topk6`, `german_csp_k6`, `french_csp_k6` (shipped on HF, fetch with
  `bash download_assets.sh`). Native `{german,french,spanish}_r32` re-exportable from the DGX checkpoints
  (`reference_dgx_ft_checkpoints`) — NOT shipped (k6 won for DE/FR).
- steer vectors (`presets/steer/emotion/ryan_*.qlsteer`): committed in git.

## Try it
`make emotion-demo` (Italian ×6 + multilingual + galatea clone) → `samples/tests/emotion_demo/`.
