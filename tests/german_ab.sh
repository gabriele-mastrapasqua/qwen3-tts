#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/german_ab; mkdir -p "$OUT"
TXT="Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."
R32=presets/expr/german_r32.expr; R64=presets/expr/german_r64.expr
C="-l German -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [neutral]=""
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
  [anger]="Speak with intense, heated, furious anger."
)
gen() { if [ -z "$2" ]; then ./qwen_tts $1 $C --text "$TXT" -o "$3" 2>/dev/null
        else ./qwen_tts $1 $C --instruct "$2" --text "$TXT" -o "$3" 2>/dev/null; fi; }

echo "### PRESET vivian (base vs +german_r32)"
for emo in neutral sad anger; do
  gen "-d qwen3-tts-1.7b -s vivian"            "${INST[$emo]}" "$OUT/viv_base_$emo.wav"
  gen "-d qwen3-tts-1.7b -s vivian --expr $R32" "${INST[$emo]}" "$OUT/viv_r32_$emo.wav"
  echo "  viv $emo done"
done
echo "### CLONE galatea cross-lingual German (base vs +german_r64)"
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
for emo in sad anger; do
  gen "-d qwen3-tts-1.7b $GAL"            "${INST[$emo]}" "$OUT/gal_base_$emo.wav"
  gen "-d qwen3-tts-1.7b $GAL --expr $R64" "${INST[$emo]}" "$OUT/gal_r64_$emo.wav"
  echo "  gal $emo done"
done
echo "DONE -> $OUT"
