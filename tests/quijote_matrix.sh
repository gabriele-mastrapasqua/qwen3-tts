#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
IT=presets/expr/italian_csp_topk6.expr; D=samples/emo_retest_0622; O=samples/quijote_matrix; mkdir -p $O
GV="--load-voice voices/quijote_graft.qvoice --icl-only"
declare -A TXT=(
  [anger]="Come ti permetti di parlarmi così? Questo non lo accetto, è inaccettabile!"
  [sad]="Ho perso tutto quello che avevo, e adesso non so più cosa fare."
  [joy]="Non ci posso credere, è la notizia più bella della mia vita, sono felicissimo!"
  [fear]="C'è qualcuno in casa, ho sentito dei passi... ho paura, non so cosa fare."
  [disgust]="Ma che roba è questa? Fa davvero schifo, non riesco nemmeno a guardarla."
  [surprise]="Cosa?! Non me lo aspettavo per niente, è incredibile, sono sbalordito!")
declare -A INS=(
  [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
  [sad]="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
  [joy]="Speak with bright, radiant joy, light and warm, smiling through every word."
  [fear]="Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread."
  [disgust]="Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you."
  [surprise]="Speak with sudden astonishment and surprise, gasping and caught off guard.")
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local o="$1"; shift; ./qwen_tts -d qwen3-tts-1.7b -l Italian --seed 42 -T 1.1 "$@" -o $O/$o >/dev/null 2>&1; echo "  $o -> $(dur $O/$o)"; }
echo "=== quijote (clone ES) speaking ITALIAN — expr / steer / combine ==="
for e in anger sad joy fear disgust surprise; do
  ql=$D/ryan_${e/anger/ang}.qlsteer
  gen quijote_${e}_expr.wav    $GV --expr $IT --expr-weight 1.0 -I "${INS[$e]}" --text "${TXT[$e]}"
  gen quijote_${e}_steer.wav   $GV --ml-steer $ql --ml-weight 8 --ml-range 21-25 --text "${TXT[$e]}"
  gen quijote_${e}_combine.wav $GV --expr $IT --expr-weight 1.0 --ml-steer $ql --ml-weight 8 --ml-range 21-25 -I "${INS[$e]}" --text "${TXT[$e]}"
done
echo "=== DONE ==="
