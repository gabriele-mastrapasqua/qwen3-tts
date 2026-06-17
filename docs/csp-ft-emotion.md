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

# cloned voice (galatea) — graft so CV weights stay intact and the .expr applies clean:
./qwen_tts -d qwen3-tts-1.7b --load-voice voices/galatea_icl.qvoice --icl-only -l Italian -T 1.1 \
  --expr presets/expr/italian_csp.expr --expr-weight 1.2 -I "<EN instruct>" --text "..." -o out.wav
```

- **`--expr-weight` sweet spot ≈ 1.2** (presets). Above ~1.4 at T1.1 it *svaria* (rambles/over-steers);
  **T1.3 holds higher weights** (T1.1+w1.4 unstable, T1.3+w1.4 stable) — temperature lets you push weight.
- **Instruct in ENGLISH** (the 1.7B follows EN/ZH instructs, not IT). The 7 instruct strings are the same
  ones used in training (see `training/expressivity-lora/prepare_emozionalmente.py` EMO map).
- Per-emotion ear verdict: **anger sensible, sad good, joy so-so** (joy is the hardest).

### Per-emotion / per-voice weight (ear-tuned 2026-06-17 — weight is NOT one global value)
- **anger → w1.2** (stronger, more incisive, fluid) on both ryan and vivian.
- **calmer emotions (sad, etc.) → w1.0** (more precise; w1.2 over-pushes prosody).
- **ryan**: lower volume, tolerates w1.2 well; stays Italian.
- **vivian**: louder + clearer in Italian than ryan (brighter preset), BUT at **w1.2 on low-arousal emotions
  it drifts toward a Chinese accent** — vivian's weight ceiling is lower; keep vivian ≤ w1.0 except anger.
  (Theory: pushing the emotion delta hard surfaces vivian's native-CN manifold.)
- → recommended default: **w1.0 globally, bump to w1.2 only for anger** (and per-voice cap vivian).

### Known artifacts to investigate (ear, 2026-06-17)
- **Final-word lengthening** on high-arousal/fear (e.g. fear ends "oggiiii", elongated last word) at w1.2
  (and sometimes w1.0). The emotion over-extends final duration — try lower weight / check if inherent.
- **Volume mismatch**: ryan quiet vs vivian loud → loudness-normalize outputs for consistency.
- **Clone metallic** (galatea) — see refinements below (NOT clipping; graft/lite-qvoice fidelity).

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

1. **Clone metallic.** galatea has a slight metallic/chorus/reverb. Measured: **NOT clipping** (peaks
   0.34-0.44, zero saturated samples) — it's the `--icl-only` graft on the LITE 25MB qvoice, more audible
   because the clone renders ~8 dB louder (rms 0.088 vs ryan 0.034) + emotional. Try the FULL
   `galatea_17b.qvoice --icl-only`, level-normalize galatea down, and/or lower clone weight (w1.0).
2. **More force (esp. joy).** Retrain with **`--csp-layers 22,23,25,26`** (top_k=4, the 4 highest probe
   weights) and/or higher LR / more epochs → structural force without raising `--expr-weight` (avoids
   over-steer). Data+probe already in `runs/csp_italian/` → starts at the train stage.

> ⚠️ Tomorrow's retrain rule: do NOT overwrite today's win. Use a NEW output name
> (`out_csp_italian_topk4/` → `italian_csp_topk4.expr`); the scripts are committed, so any change is a tracked
> git diff. This commit is the restore point.

## Gotchas

- The Bash/dev shell may be **zsh** → unquoted `$var` does NOT word-split. Pass multi-flag args as a quoted
  ARRAY `"${arr[@]}"`, never a bare string.
- `galatea_17b.qvoice` is 3 GB → OOM-fails back-to-back on a swap-full Mac. Use the lite `galatea_icl.qvoice`
  (25 MB) + `--icl-only`, or free RAM / run one at a time.
