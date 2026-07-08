# Emotion steering vectors — VALIDATED WINS (git-tracked, tiny ~232KB each)

Multi-layer activation-steering directions injected with `--ml-steer <file> --ml-range 21-25 --ml-weight W`.
All are the **CLEAN** canonical palette (`tests/act_map_steer.py --clean` = mean-center + project-out the
neutral/energy axis → emotion-only, stops correctly, no word-lengthening). `--ml-decay 0.985` is on by default.
Built for the **1.7B** model (hidden 2048; NOT compatible with 0.6B). Ear-validated 2026-06-22 → 24 (plan_emo_v3 §7-8).

| file | voice captured | emotion | recommended weight | notes |
|---|---|---|---|---|
| `ryan_ang.qlsteer`      | ryan (preset) | anger    | — (ryan anger: prefer EXPR, steer goes metallic) | §8.2 |
| `ryan_sad.qlsteer`      | ryan | sadness  | w8 | TOP |
| `ryan_joy.qlsteer`      | ryan | joy      | w8 | |
| `ryan_fear.qlsteer`     | ryan | fear     | w4 (w8 starts metallic) | |
| `ryan_disgust.qlsteer`  | ryan | disgust  | w8 | |
| `ryan_surprise.qlsteer` | ryan | surprise | w4 | |
| `galatea_ang_ft.qlsteer`| galatea (clone, captured WITH expr) | anger | w8 | voice-native dir |
| `galatea_sad_ft.qlsteer`| galatea | sadness | w8 | |
| `vivian_ang_ft.qlsteer` | vivian (preset) | anger | w4 (drifts language → prefer EXPR for vivian) | |
| `vivian_sad_ft.qlsteer` | vivian | sadness | w8 | |

## Dyads — blended emotions (2026-07-08, ear-validated ryan EN+IT, shipped v0.12.0)
Emotion directions ADD: a weighted sum of two primary `ryan_<emo>.qlsteer` vectors renders a coherent NEW
emotion. Built with `tools/steer/dyad_mix.py OUT a.qlsteer:0.5 b.qlsteer:0.5`. Same STEER recipe (preset @
w12, L21-25). Exposed as first-class `--emotion <name>` values (and inline `[emotion]` tags).

| file | = blend of | mix | reads as |
|---|---|---|---|
| `ryan_contempt.qlsteer`    | anger + disgust   | 50/50 | sneering disdain |
| `ryan_awe.qlsteer`         | fear + surprise   | 50/50 | hushed wonder |
| `ryan_nostalgia.qlsteer`   | joy + sad         | 40/60 | bittersweet fondness (sad-lean; 50/50 too light in EN) |
| `ryan_disapproval.qlsteer` | surprise + sad    | 50/50 | let-down reproach |
| `ryan_remorse.qlsteer`     | sad + disgust     | 50/50 | guilty regret |
| `ryan_outrage.qlsteer`     | anger + surprise  | 50/50 | indignant shock |
| `ryan_despair.qlsteer`     | fear + sad        | 50/50 | hopeless dread |

Note: `joy`-paired blends over-drive on long EN carriers → mind the mix ratio.

## Shippable recipe (the weeks-long clone-emotion result, generalized galatea+quijote, IT/ES/RU/ZH/JA/KO)
- **Same/close language → STEER-clean** (max emotion): `--ml-steer <voice_emo>.qlsteer --ml-range 21-25 --ml-weight 8`.
- **Far language → COMBINE**: add `--expr presets/expr/italian_csp_topk6.expr` (the expr stabilizes/renders).
- **Anger / language-drifting voice (vivian) → EXPR carries it**; steer only as much as language tolerates (w4).
- ryan most sensitive; w12 over-steers everywhere. Source captures: `samples/emo_retest_0622/*.qamp` (raw/clean A/B variants kept there).

## How to rebuild
`QWEN_ACT_MAP=neutral.qamp ./qwen_tts ... -I "<neutral>"` + `QWEN_ACT_MAP=emo.qamp ./qwen_tts ... -I "<emotion>"`
then `python3 tests/act_map_steer.py neutral.qamp emo.qamp out.qlsteer --clean --unit-per-layer`.
