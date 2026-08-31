#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=samples/lora_spot; mkdir -p "$OUT"
TXT="Allora, lascia che ti spieghi come stanno le cose."
L=presets/expr/italian_lora.expr
C="-l Italian -T 1.1 --seed 42 -j1 --silent"
declare -A INST=(
  [neutral]=""
  [happy]="Speak with bright, warm joy, light and cheerful."
  [excited]="Speak fast and thrilled, full of excited energy."
)
gen() { # margs instruct out
  if [ -z "$2" ]; then ./qwen_tts $1 $C --text "$TXT" -o "$3" 2>/dev/null
  else ./qwen_tts $1 $C --instruct "$2" --text "$TXT" -o "$3" 2>/dev/null; fi
}
for vsel in "viv|-s vivian" "gal|--load-voice voices/galatea_graft.qvoice --icl-only"; do
  tag="${vsel%%|*}"; varg="${vsel#*|}"
  for emo in neutral happy excited; do
    gen "-d qwen3-tts-1.7b $varg"            "${INST[$emo]}" "$OUT/${tag}_base_$emo.wav"
    gen "-d qwen3-tts-1.7b $varg --expr $L"  "${INST[$emo]}" "$OUT/${tag}_lora_$emo.wav"
    echo -n "  $tag $emo base-vs-lora: "; python3 tests/compare_audio.py "$OUT/${tag}_base_$emo.wav" "$OUT/${tag}_lora_$emo.wav" 2>&1 | tail -1
  done
done
echo "DONE -> $OUT"
