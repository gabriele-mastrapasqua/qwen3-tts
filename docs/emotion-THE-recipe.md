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

## The recipe — ONE rule (ear-validated 2026-06-29, full per-language sweep)

**Pure STEER wins everywhere** — clean timbre, no noise, emotes in every language. So:

- **PRESET voice → STEER**: `presets/steer/emotion/ryan_<emo>.qlsteer` @ `--ml-range 21-25`, **`--ml-weight 12`**
  (w10 also good), `--ml-decay 0.985`, **no expr, no instruct**. The ryan CLEAN palette is used for ALL voices
  (it transfers cross-voice/cross-language).
- **CLONE voice → COMBINE**: the language `.expr` @1.0 (renders/stabilizes a cross-language clone) **+** STEER @ w12
  **+** a default English instruct. The one easy clone recipe.

**weight:** anger & fear are best at **w12**; the rest win at w10 *or* w12 → **w12 is the single default**.
(Earlier per-(voice×emotion) EXPR/COMBINE tables are SUPERSEDED — pure STEER at w12 beat them with the right speaker.)

**Use the NATIVE preset per language** (the engine applies STEER to *whatever* voice you pass, but the GOLD
voice is the language-native one; the router prints a hint):

| language | native preset | language | native preset |
|---|---|---|---|
| Japanese | **ono_anna** | Italian / English / Portuguese | **ryan** |
| Korean | **sohee** | German / French / Spanish | **vivian** (ryan also good for Romance) |
| Chinese | **vivian** / uncle_fu | (cloned voice, any language) | the clone → COMBINE |

Notes:
- `--instruct` (vivid English) and `--expr` are **optional manual overrides** on a preset — not in the default.
  COMBINE (expr+steer) gave anger/fear a "touch of class" in some languages but **broke Spanish (noise)** → it's
  the clone default, not the preset default.
- seed 42 is the reference. Temperature 1.1.

## Assets
- expr packs (`presets/expr/`): `italian_csp_topk6`, `german_csp_k6`, `french_csp_k6` (shipped on HF, fetch with
  `bash download_assets.sh`). Native `{german,french,spanish}_r32` re-exportable from the DGX checkpoints
  (`reference_dgx_ft_checkpoints`) — NOT shipped (k6 won for DE/FR).
- steer vectors (`presets/steer/emotion/ryan_*.qlsteer`): committed in git.

## Try it
`make emotion-demo` (Italian ×6 + multilingual + galatea clone) → `samples/tests/emotion_demo/`.
