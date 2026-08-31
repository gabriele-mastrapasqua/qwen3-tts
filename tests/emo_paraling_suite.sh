#!/usr/bin/env bash
set -uo pipefail
cd /Users/gabrielemastrapasqua/source/personal/qwen-tts
O=samples/emo_paraling_suite; mkdir -p $O
D=samples/emo_retest_0622
IT=presets/expr/italian_csp_topk6.expr
MODE="${MODE:-combine}"
SEEDS=(${SEEDS:-42 777 123})
SPEAKERS=(${SPEAKERS:-ryan vivian serena galatea})
EMOS=(
  "joy|ahah|Non ci posso credere, ahah, e la notizia piu bella della mia vita.|joy"
  "sad|ahh|Ho perso tutto quello che avevo, ahh, e adesso non so piu cosa fare.|sad"
  "anger|grr|Come ti permetti di parlarmi cosi, grr, questo proprio non lo accetto.|ang"
)
declare -A INS=(
  [joy]="Speak with bright, radiant joy, light and warm, laughing and smiling through the words."
  [sad]="Speak in a sad, sorrowful, gloomy tone, voice low and heavy, sighing, on the verge of tears."
  [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, growling, barely holding back the rage."
)
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in
  galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";;
  *)       echo "-s $1";;
esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }   # expr-weight: clone 1.0, preset 1.2

gen(){ # gen <speaker> <emo> <word> <sentence> <steertag> <seed>
  local sp="$1" emo="$2" word="$3" txt="$4" stag="$5" seed="$6"
  local out="${sp}_${emo}_${word}_${MODE}_s${seed}.wav"
  local steer="$D/ryan_${stag}.qlsteer"
  local args=(-d qwen3-tts-1.7b $(vf $sp) -l Italian --seed $seed -T 1.1)
  case "$MODE" in
    combine) args+=(--expr $IT --expr-weight $(ew $sp) --ml-steer $steer --ml-weight 8 --ml-range 21-25 -I "${INS[$emo]}");;
    steer)   args+=(--ml-steer $steer --ml-weight 8 --ml-range 21-25);;
    expr)    args+=(--expr $IT --expr-weight $(ew $sp) -I "${INS[$emo]}");;
  esac
  ./qwen_tts "${args[@]}" --text "$txt" -o $O/$out >/dev/null 2>&1
  echo "  $out -> $(dur $O/$out)"
}

echo "===== EMO+PARALING SUITE  mode=$MODE  speakers=[${SPEAKERS[*]}]  seeds=[${SEEDS[*]}] ====="
for sp in "${SPEAKERS[@]}"; do
  for e in "${EMOS[@]}"; do
    IFS='|' read -r emo word txt stag <<< "$e"
    echo "--- $sp / $emo ($word) ---"
    for s in "${SEEDS[@]}"; do gen "$sp" "$emo" "$word" "$txt" "$stag" "$s"; done
  done
done
echo "===== DONE -> $O ====="
