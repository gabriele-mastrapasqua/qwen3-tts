#!/usr/bin/env bash
set -uo pipefail
cd /Users/gabrielemastrapasqua/source/personal/qwen-tts
O=samples/paraling_compare; mkdir -p $O
D=samples/emo_retest_0622
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in ryan) echo "-s ryan";; vivian) echo "-s vivian";; galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; esac; }
gen(){ local out="$1" v="$2" txt="$3"; shift 3
  local steer=""; [ $# -ge 2 ] && steer="--ml-steer $1 --ml-weight $2 --ml-range 21-25"
  ./qwen_tts -d qwen3-tts-1.7b $(vf $v) -l Italian --seed 42 -T 1.1 $steer --text "$txt" -o $O/$out >/dev/null 2>&1
  echo "  $out -> $(dur $O/$out)"; }

LAUGH_AH="Che bella notizia, ahah, non ci posso proprio credere."
LAUGH_EH="Che bella notizia, eheh, non ci posso proprio credere."
SIGH_UFF="Che giornata, uff, sono davvero stanco e non ce la faccio piu."
SIGH_AHH="Che giornata, ahh, sono davvero stanco e non ce la faccio piu."

for v in ryan vivian galatea; do
  echo "===== $v ====="
  gen ${v}_laugh_V1ahah.wav       $v "$LAUGH_AH"
  gen ${v}_laugh_V1ahah_joy.wav   $v "$LAUGH_AH" $D/ryan_joy.qlsteer 8
  gen ${v}_laugh_V1eheh_joy.wav   $v "$LAUGH_EH" $D/ryan_joy.qlsteer 8
  gen ${v}_sigh_V1uff.wav         $v "$SIGH_UFF"
  gen ${v}_sigh_V1uff_sad.wav     $v "$SIGH_UFF" $D/ryan_sad.qlsteer 8
  gen ${v}_sigh_V1ahh_sad.wav     $v "$SIGH_AHH" $D/ryan_sad.qlsteer 8
done
echo "===== V1-word DONE ====="
