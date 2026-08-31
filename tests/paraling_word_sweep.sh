#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/paraling_sweep; mkdir -p $O
D=samples/emo_retest_0622
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in ryan) echo "-s ryan";; vivian) echo "-s vivian";; galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; esac; }
gen(){ local out="$1" v="$2" emo="$3" txt="$4"; ./qwen_tts -d qwen3-tts-1.7b $(vf $v) -l Italian --seed 42 -T 1.1 --ml-steer $D/ryan_${emo}.qlsteer --ml-weight 8 --ml-range 21-25 --text "$txt" -o $O/$out >/dev/null 2>&1; echo "  $out -> $(dur $O/$out)"; }

declare -a LW=( "ahahah:Che bella notizia, ahahah, non ci posso credere."
                "hahaha:Che bella notizia, hahaha, non ci posso credere."
                "ahahahah:Che bella notizia, ahahahah, non ci posso credere."
                "zhCN:Che bella notizia, 哈哈哈, non ci posso credere."
                "ihih:Che bella notizia, ihihih, non ci posso credere." )
declare -a SW=( "ahh:Che giornata, ahh, sono stanco e non ce la faccio piu."
                "ahhh:Che giornata, ahhh, sono stanco e non ce la faccio piu."
                "uff:Che giornata, uff, sono stanco e non ce la faccio piu."
                "zhSigh:Che giornata, 唉, sono stanco e non ce la faccio piu." )

for v in ryan vivian galatea; do
  echo "===== $v — LAUGH (joy) ====="
  for item in "${LW[@]}"; do id="${item%%:*}"; txt="${item#*:}"; gen ${v}_laugh_${id}.wav $v joy "$txt"; done
  echo "===== $v — SIGH (sad) ====="
  for item in "${SW[@]}"; do id="${item%%:*}"; txt="${item#*:}"; gen ${v}_sigh_${id}.wav $v sad "$txt"; done
done
echo "===== SWEEP DONE -> $O ====="
