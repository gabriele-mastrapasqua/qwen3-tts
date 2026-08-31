#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
D=samples/emo_retest_0622; O=samples/paraling_mix; mkdir -p $O
GV="--load-voice voices/galatea_graft.qvoice --icl-only"
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local o="$1"; shift; ./qwen_tts -d qwen3-tts-1.7b -l Italian --seed 42 -T 1.1 "$@" -o $O/$o >/dev/null 2>&1; echo "  $o -> $(dur $O/$o)"; }
steer(){ local emo="$1"; echo "$D/ryan_${emo}.qlsteer"; }

echo "===== JOY (+ laughter) ====="
gen joy_A_leveralone.wav  $GV --ml-steer $(steer joy) --ml-weight 8 --ml-range 21-25 --text "Non ci posso credere, è la notizia più bella della mia vita!"
gen joy_B_blended.wav     $GV --ml-steer $(steer joy) --ml-weight 8 --ml-range 21-25 --text "Ahahah! Non ci posso credere, è la notizia più bella della mia vita!"
gen joy_C_macro.wav       $GV --ml-steer $(steer joy) --ml-weight 8 --ml-range 21-25 --text "Non ci posso credere, è la notizia più bella della mia vita! [laugh]"

echo "===== ANGER (+ growl) ====="
gen anger_A_leveralone.wav $GV --ml-steer $(steer ang) --ml-weight 8 --ml-range 21-25 --text "Come ti permetti di parlarmi così? Questo non lo accetto!"
gen anger_B_blended.wav    $GV --ml-steer $(steer ang) --ml-weight 8 --ml-range 21-25 --text "Grrr! Come ti permetti di parlarmi così? Questo non lo accetto!"

echo "===== DISGUST (+ bleah) ====="
gen disgust_A_leveralone.wav $GV --ml-steer $(steer disgust) --ml-weight 8 --ml-range 21-25 --text "Ma che roba è questa? Fa davvero schifo."
gen disgust_B_blended.wav    $GV --ml-steer $(steer disgust) --ml-weight 8 --ml-range 21-25 --text "Bleah! Ma che roba è questa? Fa davvero schifo."

echo "===== SAD (+ sigh/moan) ====="
gen sad_A_leveralone.wav $GV --ml-steer $(steer sad) --ml-weight 8 --ml-range 21-25 --text "Ho perso tutto quello che avevo, e adesso non so più cosa fare."
gen sad_B_blended.wav    $GV --ml-steer $(steer sad) --ml-weight 8 --ml-range 21-25 --text "Ahimè... ho perso tutto quello che avevo... uff... non so più cosa fare."
gen sad_C_macro.wav      $GV --ml-steer $(steer sad) --ml-weight 8 --ml-range 21-25 --text "[sigh] Ho perso tutto quello che avevo, e adesso non so più cosa fare."
echo "===== DONE ====="
