#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/lora_rank; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
R16=presets/expr/italian_lora.expr
R32=presets/expr/italian_lora_r32.expr
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [neutral]=""
  [sad]="Speak deeply sad and heartbroken, a slow broken voice on the verge of tears."
  [anger]="Speak with intense, heated, furious anger."
)
gen() { if [ -z "$2" ]; then ./qwen_tts $1 $C --text "$TXT" -o "$3" 2>/dev/null
        else ./qwen_tts $1 $C --instruct "$2" --text "$TXT" -o "$3" 2>/dev/null; fi; }
for vsel in "viv|-s vivian" "gal|--load-voice voices/galatea_graft.qvoice --icl-only"; do
  tag="${vsel%%|*}"; varg="${vsel#*|}"
  for emo in neutral sad anger; do
    gen "-d qwen3-tts-1.7b $varg --expr $R16" "${INST[$emo]}" "$OUT/${tag}_r16_$emo.wav"
    gen "-d qwen3-tts-1.7b $varg --expr $R32" "${INST[$emo]}" "$OUT/${tag}_r32_$emo.wav"
    echo -n "  $tag $emo r16-vs-r32: "; python3 tests/compare_audio.py "$OUT/${tag}_r16_$emo.wav" "$OUT/${tag}_r32_$emo.wav" 2>&1 | tail -1
  done
done
echo "DONE -> $OUT"
