# Paralinguistic steering vectors — VALIDATED WINS (git-tracked, tiny ~232KB each)

Injectable, **speaker- AND language-agnostic** activation directions for VOCAL paralinguistic events.
Inject with `--ml-steer <file> --ml-range 21-25 --ml-weight W`. Built on the **1.7B** model (hidden 2048).

⚠️ **Two rules opposite to emotion steering** (energy IS the signal here):
1. Built **RAW** (`--unit-per-layer`, **NO `--clean`**) — project-out-energy would collapse a laugh into a cry.
2. Contrast against the **CONFUSABLE OPPOSITE**, not neutral (laugh−cry, sigh−laugh).

| file | = (event − opposite) | recommended weight | ear verdict |
|---|---|---|---|
| `laugh_vs_cry.qlsteer` | ryan-EN laugh − ryan-EN cry | galatea 8 · vivian 8 · ryan 6 | top result |
| `sigh_vs_laugh.qlsteer` | ryan-EN sigh − ryan-EN laugh | galatea 8 · vivian 8 · ryan 6 | win |

Source captures to rebuild: `ryan_en_laugh.qamp`, `ryan_en_sigh.qamp` (cry capture in `samples/para_steer_vec/`).

## ⭐ KEY RECIPE RULE (discovered 2026-06-28)
A para vector ALONE gives only the prosody, **NOT** the discrete event. It needs a **native-trigger
onomatopoeia in the carrier text** to seed the event — the vector then SHAPES/reinforces it:
- laugh → carrier contains `ahah`/`hahaha`
- sigh  → carrier contains `haaah`
On top of the normal COMBINE (`--expr presets/expr/italian_csp_topk6.expr` + emotion instruct + T1.1).
w12 over-steers everywhere; ryan caps at w6.

## Scope
VOCAL family only (laugh, sigh). Articulatory events (cough/sneeze/disgust/gasp/growl) FAIL via steering
(decoder ceiling) → use native-trigger onomatopoeia inline (`ugh`/`ahem`/`tsk`/`haaa`) instead.

## How to rebuild
`act_map_steer.py <opposite>.qamp <event>.qamp out.qlsteer --unit-per-layer` (RAW, NO --clean).
