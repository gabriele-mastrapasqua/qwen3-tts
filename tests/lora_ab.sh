#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/lora_ab; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
BASE=qwen3-tts-1.7b
A=presets/expr/italian.expr
L=presets/expr/italian_lora.expr
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [anger]="Speak with intense, heated, furious anger."
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
)
gen() { ./qwen_tts $1 $C --instruct "$2" --text "$TXT" -o "$3" 2>/dev/null; }

for vsel in "viv|-s vivian" "gal|--load-voice voices/galatea_graft.qvoice --icl-only"; do
  tag="${vsel%%|*}"; varg="${vsel#*|}"
  for emo in sad anger; do
    gen "-d $BASE $varg"            "${INST[$emo]}" "$OUT/${tag}_base_$emo.wav"
    gen "-d $BASE $varg --expr $A"  "${INST[$emo]}" "$OUT/${tag}_rta_$emo.wav"
    gen "-d $BASE $varg --expr $L"  "${INST[$emo]}" "$OUT/${tag}_lora_$emo.wav"
    echo -n "  $tag $emo  LoRA-vs-routeA: "; python3 tests/compare_audio.py "$OUT/${tag}_rta_$emo.wav" "$OUT/${tag}_lora_$emo.wav" 2>&1 | tail -1
  done
done
echo "DONE -> $OUT"
