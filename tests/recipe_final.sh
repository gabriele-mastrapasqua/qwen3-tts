#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42; T=1.1
IT=presets/expr/italian_csp_topk6.expr
QL=samples/emo_retest_0622   # ryan_<emo>.qlsteer are now CLEAN canonical
O=samples/recipe_final; mkdir -p $O
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
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
ql(){ echo "$QL/ryan_${1/anger/ang}.qlsteer"; }
g(){ local out="$1"; shift; $BIN -d $M -l Italian --seed $SEED -T $T "$@" -o "$O/$out" >/dev/null 2>&1; echo "  $out -> $(dur "$O/$out")"; }
EXPR(){ local vf="$1" e="$2" ew="$3" o="$4"; g "$o" $vf --expr $IT --expr-weight $ew -I "${INS[$e]}" --text "${TXT[$e]}"; }
STEER(){ local vf="$1" e="$2" w="$3" o="$4"; g "$o" $vf --ml-steer "$(ql $e)" --ml-weight $w --ml-range 21-25 --text "${TXT[$e]}"; }
COMBINE(){ local vf="$1" e="$2" ew="$3" w="$4" o="$5"; g "$o" $vf --expr $IT --expr-weight $ew --ml-steer "$(ql $e)" --ml-weight $w --ml-range 21-25 -I "${INS[$e]}" --text "${TXT[$e]}"; }

echo "===== RYAN (preset) ====="
EXPR    "-s ryan" anger    1.2     ryan_anger_EXPR.wav
COMBINE "-s ryan" disgust  1.2 8   ryan_disgust_COMBINE.wav
STEER   "-s ryan" fear     4       ryan_fear_STEER.wav
COMBINE "-s ryan" joy      1.2 8   ryan_joy_COMBINE.wav
STEER   "-s ryan" sad      8       ryan_sad_STEER.wav
STEER   "-s ryan" surprise 4       ryan_surprise_STEER.wav

echo "===== VIVIAN (preset, drifts → expr for language) ====="
EXPR    "-s vivian" anger    1.2     vivian_anger_EXPR.wav
STEER   "-s vivian" disgust  4       vivian_disgust_STEER.wav
STEER   "-s vivian" fear     8       vivian_fear_STEER.wav
COMBINE "-s vivian" joy      1.2 8   vivian_joy_COMBINE.wav
STEER   "-s vivian" sad      8       vivian_sad_STEER.wav
STEER   "-s vivian" surprise 8       vivian_surprise_STEER.wav

echo "===== GALATEA — clone path A: qvoice 3GB (--icl-only) ====="
QV="--load-voice voices/galatea_graft.qvoice --icl-only"
STEER   "$QV" anger    8       galatea_anger_STEER_qvoice.wav
COMBINE "$QV" anger    1.0 8   galatea_anger_COMBINE_qvoice.wav
COMBINE "$QV" disgust  1.0 8   galatea_disgust_COMBINE_qvoice.wav
COMBINE "$QV" fear     1.0 8   galatea_fear_COMBINE_qvoice.wav
STEER   "$QV" joy      8       galatea_joy_STEER_qvoice.wav
STEER   "$QV" sad      8       galatea_sad_STEER_qvoice.wav
STEER   "$QV" surprise 8       galatea_surprise_STEER_qvoice.wav

echo "===== GALATEA — clone path B: galatea.bin 8KB (--xvector-only) — the WIN-WIN test ====="
BN="--load-voice voices/galatea.bin --xvector-only"
STEER   "$BN" anger    8       galatea_anger_STEER_bin.wav
COMBINE "$BN" anger    1.0 8   galatea_anger_COMBINE_bin.wav
COMBINE "$BN" disgust  1.0 8   galatea_disgust_COMBINE_bin.wav
COMBINE "$BN" fear     1.0 8   galatea_fear_COMBINE_bin.wav
STEER   "$BN" joy      8       galatea_joy_STEER_bin.wav
STEER   "$BN" sad      8       galatea_sad_STEER_bin.wav
STEER   "$BN" surprise 8       galatea_surprise_STEER_bin.wav
echo "===== recipe_final DONE ====="
