#!/usr/bin/env bash
# A/B the layer BAND: emotion-only L16-26 vs broad early+emotion (0-27), same EMOVO data.
# Tests both goals (docs/prosody-map.md hypothesis): better EMOTION + better Italian RENDERING.
# Usage: tests/lora_band_ab.sh   (edit BASE/BB paths below if needed)
set -uo pipefail
D=qwen3-tts-1.7b; SEED=42; T=1.1; L=Italian
BASE=presets/expr/italian_r32.expr           # L16-26 (yesterday)
BB=presets/expr/italian_bb027_r32.expr       # L00-27 (broad band)
GAL=voices/galatea_17b.qvoice
O=/tmp/lora_band_ab; mkdir -p "$O"
EMO="Non posso credere che tu l'abbia fatto davvero, sono fuori di me!"   # emotional line
NEU="Domani mattina prendo il treno delle otto per andare in citta."      # neutral: rendering+phonetics
INSTR="Speak with hot, furious anger, sharp and forceful."
run() { echo "+ $*"; ./qwen_tts -d $D --seed $SEED -T $T -l $L "$@"; }

echo "===== VIVIAN (Chinese-native preset -> IT): emotional ====="
run -s vivian --text "$EMO" --instruct "$INSTR"            -o $O/vivian_emo_base.wav
run -s vivian --text "$EMO" --instruct "$INSTR" --expr $BASE -o $O/vivian_emo_L1626.wav
run -s vivian --text "$EMO" --instruct "$INSTR" --expr $BB   -o $O/vivian_emo_bb.wav
echo "===== VIVIAN: neutral (rendering + phonetics check) ====="
run -s vivian --text "$NEU"                 -o $O/vivian_neu_base.wav
run -s vivian --text "$NEU" --expr $BASE    -o $O/vivian_neu_L1626.wav
run -s vivian --text "$NEU" --expr $BB      -o $O/vivian_neu_bb.wav

echo "===== RYAN: emotional head-to-head ====="
run -s ryan --text "$EMO" --instruct "$INSTR" --expr $BASE -o $O/ryan_emo_L1626.wav
run -s ryan --text "$EMO" --instruct "$INSTR" --expr $BB   -o $O/ryan_emo_bb.wav
echo "===== RYAN: neutral head-to-head ====="
run -s ryan --text "$NEU" --expr $BASE -o $O/ryan_neu_L1626.wav
run -s ryan --text "$NEU" --expr $BB   -o $O/ryan_neu_bb.wav

echo "===== GALATEA (clone): broad band r32 — does it work on a clone? ====="
run --load-voice $GAL --icl-only --text "$EMO" --instruct "$INSTR" --expr $BB -o $O/galatea_emo_bb.wav
run --load-voice $GAL --icl-only --text "$NEU" --expr $BB                      -o $O/galatea_neu_bb.wav

echo; echo "DONE -> $O/  (BASE=L16-26  BB=0-27)"
echo "Listen head-to-head: *_L1626 vs *_bb  (emotion stronger? Italian cleaner? phonetics OK?)"
