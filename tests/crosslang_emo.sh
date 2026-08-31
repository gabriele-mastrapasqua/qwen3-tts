#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
IT=presets/expr/italian_csp_topk6.expr; D=samples/emo_retest_0622; O=samples/crosslang_emo; mkdir -p $O
GV="--load-voice voices/galatea_graft.qvoice --icl-only"
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local o="$1"; shift; ./qwen_tts -d qwen3-tts-1.7b --seed 42 -T 1.1 "$@" -o $O/$o >/dev/null 2>&1; echo "  $o -> $(dur $O/$o)"; }

declare -A INS=(
  [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
  [sad]="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
  [joy]="Speak with bright, radiant joy, light and warm, smiling through every word.")

LANGS=(Russian Chinese Japanese Korean)
declare -A TXT=(
  [Russian_anger]="Как ты смеешь так со мной разговаривать? Это неприемлемо, я этого не потерплю!"
  [Russian_sad]="Я потерял всё, что у меня было, и теперь я не знаю, что мне делать."
  [Russian_joy]="Не могу поверить, это лучшая новость в моей жизни, я так счастлив!"
  [Chinese_anger]="你怎么敢这样跟我说话？这我无法接受，太过分了！"
  [Chinese_sad]="我失去了我所拥有的一切，现在我不知道该怎么办。"
  [Chinese_joy]="我简直不敢相信，这是我一生中最好的消息，我太高兴了！"
  [Japanese_anger]="よくも私にそんな口のきき方ができるな！こんなのは絶対に受け入れられない！"
  [Japanese_sad]="私が持っていたものを全て失って、もうどうすればいいのか分からない。"
  [Japanese_joy]="信じられない、人生で一番いい知らせだ、本当に幸せだ！"
  [Korean_anger]="네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!"
  [Korean_sad]="내가 가진 모든 걸 잃었어, 이제 어떻게 해야 할지 모르겠어."
  [Korean_joy]="믿을 수가 없어, 내 인생 최고의 소식이야, 정말 너무 행복해!")

for L in "${LANGS[@]}"; do
  echo "===== $L ====="
  for e in anger sad joy; do
    t="${TXT[${L}_${e}]}"; ql=$D/ryan_${e/anger/ang}.qlsteer
    gen ${L}_${e}_steer.wav   $GV -l $L --ml-steer $ql --ml-weight 8 --ml-range 21-25 --text "$t"
    gen ${L}_${e}_combine.wav $GV -l $L --expr $IT --expr-weight 1.0 --ml-steer $ql --ml-weight 8 --ml-range 21-25 -I "${INS[$e]}" --text "$t"
  done
done
echo "===== crosslang DONE ====="
