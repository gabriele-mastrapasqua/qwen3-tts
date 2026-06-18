# CSP-FT Italian Emotion — the WIN recipe (2026-06-17)

Emotion that **generalizes to cloned voices** with **clean pronunciation**, via Characteristic-Specific
Partial Fine-Tuning (CSP-FT, arXiv 2501.14273). First time emotion moves correctly on BOTH presets and a
cloned `.qvoice`, in Italian, with no word/length glitches. Ear-validated.

## The recipe (use this)

```bash
# preset (ryan / vivian), Italian, emotion via EN instruct + the CSP .expr:
./qwen_tts -d qwen3-tts-1.7b -s ryan -l Italian -T 1.1 \
  --expr presets/expr/italian_csp.expr --expr-weight 1.2 \
  -I "Speak with hot, furious anger, sharp and forceful." \
  --text "Non posso credere a quello che è successo oggi." -o anger.wav

# cloned voice — DEFAULT = x-vector-only from a tiny 8KB .bin (clean, no room-reverb, more force headroom):
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea.bin --xvector-only -l Italian -T 1.3 \
  --expr presets/expr/italian_csp.expr --expr-weight 2.0 -I "<EN instruct>" --text "..." -o out.wav
```

> Make the 8 KB `.bin` once from any `.qvoice`: `python3 tests/qvoice_to_xvec.py voices/yourvoice.qvoice`
> (or clone straight to a `.bin`: `--ref-audio ref24k.wav --xvector-only --save-voice voices/yourvoice.bin`).

**Why x-vector-only is the clone default (ear-validated 2026-06-18):** a `.qvoice` ICL clone carries
`ref_codes` = the codec of the reference RECORDING, which re-injects that recording's room acoustics (a
"muffled metallic / faint reverb") into every generation — and the CSP `.expr` makes it MORE audible (it
re-attends the ref_codes harder). The x-vector carries abstract identity WITHOUT the room → glitch gone,
**identity holds**, and the clone takes **much higher weight** (the ICL amplification path is removed, so you
push the `.expr` harder for the same emotional movement). See `docs/icl-graft-portability.md` / plan_emo_v2.md.

- **Preset defaults: `--expr-weight ≈ 1.2`, T1.1.** Above ~1.4 at T1.1 it *svaria* (over-steers); **T1.3 holds
  higher weights** — temperature lets you push weight.
- **x-vector clone defaults: `-T 1.3`, per-emotion weight ~1.6–2.0** (vs preset/ICL ~1.0–1.2). x-vector needs
  more weight (no ICL amplification) and tolerates it. **w2.5/T1.3 svaria** (loses Italian, random words).
- **Instruct in ENGLISH** (the 1.7B follows EN/ZH instructs, not IT). The 7 instruct strings are the same
  ones used in training (see `training/expressivity-lora/prepare_emozionalmente.py` EMO map).
- Per-emotion ear verdict: **anger TOP (x-vector w2.0/T1.3), sad good, joy/disgust still weak** (joy/disgust
  do NOT unlock with weight → training ceiling → retrain top_k=4, see refinements).

### Per-emotion / per-voice weight (ear-tuned — weight is NOT one global value)
**x-vector clone @ T1.3:** anger ~2.0 · sad ~1.2 · fear/surprise ~1.6 · disgust ~1.8 (still flat) · joy ~1.6
(still flat). **Preset @ T1.1:** anger w1.2; calmer emotions w1.0.
- **ryan** (preset): lower volume, tolerates w1.2 well; stays Italian.
- **vivian** (preset): louder + clearer in IT, BUT at **w1.2 on low-arousal it drifts to a Chinese accent** —
  cap vivian ≤ w1.0 except anger.
- → preset default: **w1.0 globally, bump anger to w1.2**. x-vector-clone default: **T1.3 + per-emotion above**.

### Clone fidelity tiers — pick by need (nothing is deprecated)
| Tier | How | Size | Use when |
|------|-----|------|----------|
| **x-vector `.bin`** — DEFAULT | `--load-voice X.bin --xvector-only` | **8 KB** | expr/emotion, clean output, small files (this recipe) |
| **ICL graft** | `--load-voice X_icl.qvoice --icl-only` | ~24 MB | a bit more timbre mimicry, ref is studio-clean (carries the room reverb the `.expr` amplifies) |
| **full WDELTA** | `--load-voice X.qvoice` (no `--icl-only`) | ~0.8–3 GB | **1:1 fidelity guarantee** — the heaviest, most faithful clone; keep for when you need exact identity |

The x-vector `.bin` is the default; the ICL and full-WDELTA paths are kept (not deprecated) for higher-fidelity
needs. Note: the dense `.expr` bit-delta applies cleanly only on CV-intact weights (preset / `--xvector-only` /
`--icl-only`); on a full-WDELTA clone it garbles → for emotion on a clone, use the x-vector default.

### Known artifacts to investigate (ear)
- **Clone room-reverb metallic — ✅ FIXED** by the x-vector-only default (was the ICL ref_codes; see above).
- **Final-word lengthening** on high-arousal/fear ("oggiiii") at high weight — check if it persists on x-vector.
- **Volume mismatch**: ryan quiet vs vivian loud, clone ~8 dB louder → loudness-normalize outputs.

## How it was built

1. **Data** (`prepare_emozionalmente.py`): EMOVO (0.5h, 6 pro actors) + **Emozionalmente** (~6h, 431 actors,
   Zenodo 12616095) → ~6.5h Italian emotional, single-language (no cross-language negative transfer). Used the
   human-validation CSVs: **smart-agreement** (keep ≥3/5 human-recognized for anger/joy/sad/surprise/neutral,
   keep ALL for the intrinsically-ambiguous disgust/fear) + `--loudnorm` (level fix, no data loss). 5770 clips.
2. **Probe** (`csp_probe.py`): a softmax-weighted layer probe + emotion classifier on the frozen Talker finds
   **emotion concentrates LATE: L22-27, peak L22-23** (weights L22 .110 > L23 .084 > L25 > L26 > L24 > L27).
   This refines the old hand-picked L16-26 (the real core is later/tighter). Selected blocks = **[22, 23]**.
3. **CSP-FT** (`dgx_sft_expr_csp.py`): fine-tune ONLY blocks 22+23 (whole block, `--scope full`), FREEZE
   everything else incl. pronunciation → speech stays clean (no "gomani"). 10 epochs. Output = a full CV
   checkpoint; `tests/expr_extract.py` extracts the bit-delta vs base → `presets/expr/italian_csp.expr` (72MB,
   only the 2 blocks). Train-loss stays ~7 (flat) — EXPECTED with 92% frozen; verdict is ear, not loss.
4. **Orchestrator** (`dgx_csp_italian.sh`): runs all of the above on the DGX, ISOLATED in `runs/csp_italian/`
   (own data/markers/outputs — never collides with other runs). `SMART=1 TRAIN_JUDGE=1 nohup bash dgx_csp_italian.sh`.

## Reproduce the .expr from the checkpoint

```bash
scp -r dgx:qwen-ft/runs/csp_italian/out_csp_italian/checkpoint-final /tmp/csp_ckpt
python3 tests/expr_extract.py qwen3-tts-1.7b /tmp/csp_ckpt presets/expr/italian_csp.expr --lang Italian
```

## SER judge (objective testing) — works on humans, NOT on TTS

`train_ser_judge.py` trains a 7-class Italian SER on Emozionalmente (speaker-independent split): **TEST UAR
0.789** (human baseline 66%). `emo_judge.py` scores a dir of clips → recognizability table. **Caveat
(measured):** the judge nails REAL Emozionalmente clips (8/8) but is OUT-OF-DOMAIN on our synthetic TTS audio
(scores ~0/7 even on the working preset recipe) → **do NOT trust its UAR on TTS; the verdict is the EAR.** To
make it a TTS referee it would need domain-adaptation (fine-tune on TTS-generated audio). Spin-off idea: a
standalone `italian-ser-judge` repo (MIT for code; Emozionalmente is cite-only — Catania et al., IEEE TASLP
33:1142-1155, 2025, doi:10.1109/TASLPRO.2025.3540662).

## Open refinements (TODO — see plan_emo_v2.md)

1. **Clone metallic — ✅ RESOLVED 2026-06-18.** It was the **ICL ref_codes** (the reference recording's room
   reverb), NOT clipping and NOT embedding quality (the two galatea qvoices have byte-identical x-vectors).
   Fix shipped: **x-vector-only `.bin` is now the clone default** (see recipe above). The 2.8GB WDELTA file is
   not needed for this path.
2. **More force (esp. joy/disgust).** Retrain with **`--csp-layers 22,23,25,26`** (top_k=4, the 4 highest probe
   weights) and/or higher LR / more epochs → structural force without raising `--expr-weight` (avoids
   over-steer). Data+probe already in `runs/csp_italian/` → starts at the train stage.

> ⚠️ Tomorrow's retrain rule: do NOT overwrite today's win. Use a NEW output name
> (`out_csp_italian_topk4/` → `italian_csp_topk4.expr`); the scripts are committed, so any change is a tracked
> git diff. This commit is the restore point.

## Gotchas

- The Bash/dev shell may be **zsh** → unquoted `$var` does NOT word-split. Pass multi-flag args as a quoted
  ARRAY `"${arr[@]}"`, never a bare string.
- `galatea_17b.qvoice` is 3 GB → OOM-fails back-to-back on a swap-full Mac. The clone default is the **8 KB
  `voices/galatea.bin` + `--xvector-only`** (no OOM, no room-reverb); the big WDELTA `.qvoice` is not needed.
