#!/usr/bin/env bash
cd "$(dirname "$0")/.."
OUT=samples/speaker_matrix; mkdir -p "$OUT"
M=qwen3-tts-1.7b
TXT="Allora, lascia che ti spieghi come stanno le cose."
SEED=42; T=1.1
ANGRY="Speak angrily, sharp and confrontational, clearly upset."
SAD="Speak with a sad, sorrowful, downcast tone, voice low and heavy."
EXCITED="Speak with bright, bubbling excitement, fast and energetic."

run(){ # $1=outname $2=instruct(may be empty) $3..=voice args
  local out="$OUT/$1.wav"; local instr="$2"; shift 2
  if [ -z "$instr" ]; then
    timeout 90 ./qwen_tts -d "$M" "$@" --seed $SEED -T $T --max-duration 16 -l Italian \
      --text "$TXT" -o "$out" --silent 2>/dev/null
  else
    timeout 90 ./qwen_tts -d "$M" "$@" --seed $SEED -T $T --max-duration 16 -l Italian \
      --instruct "$instr" --text "$TXT" -o "$out" --silent 2>/dev/null
  fi
  local d=$(python3 -c "import os;print(f'{os.path.getsize(\"$out\")/2/24000:.1f}')" 2>/dev/null)
  echo "    $1 -> ${d}s"
}

voice_block(){ # $1=name  $2..=voice args
  local name="$1"; shift
  run "${name}__neutral" "" "$@"
  run "${name}__angry"   "$ANGRY" "$@"
  run "${name}__sad"     "$SAD" "$@"
  run "${name}__excited" "$EXCITED" "$@"
  echo "  done $name"
}

for VN in serena vivian uncle_fu ryan aiden ono_anna sohee eric dylan; do
  voice_block "$VN" -s "$VN"
done
voice_block galatea --load-voice voices/galatea_graft.qvoice --icl-only
echo "ALL DONE -> $OUT ($(ls $OUT/*.wav 2>/dev/null|wc -l|tr -d ' ') files)"
