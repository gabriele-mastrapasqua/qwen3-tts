#!/usr/bin/env bash
set -uo pipefail
cd /Users/gabrielemastrapasqua/source/personal/qwen-tts
O=samples/para_cn_vec; rm -rf $O; mkdir -p $O
M=qwen3-tts-1.7b; IT=presets/expr/italian_csp_topk6.expr; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
vf(){ case "$1" in galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";; *) echo "-s $1";; esac; }
ew(){ [ "$1" = galatea ] && echo 1.0 || echo 1.2; }
wt(){ [ "$1" = ryan ] && echo 6 || echo 8; }

EVENTS=(
"gasp|突然受惊倒吸一口气，震惊地喘息，难以置信。|啊，我简直不敢相信，哎呀，太意外了！|缓缓地长长地叹气，气息缓缓呼出，疲惫而无奈。|唉，今天真累，唉，我撑不下去了。|Oh, non ci posso credere, non me lo aspettavo per niente!"
"growl|用愤怒咆哮的语气，从胸腔发出低沉的怒吼，咬牙切齿，怒不可遏。|你竟然敢这样对我，哼，我绝对不能接受。|用温柔轻柔、亲切安抚的语气说话，平静而温暖。|没关系，慢慢来，一切都会好的，深呼吸。|Come ti permetti di parlarmi cosi, questo proprio non lo accetto."
"disgust|用强烈厌恶、恶心反胃的语气，在词语之间发出不由自主的干呕和嫌弃的声音。|呸，这个东西太恶心了，呕，我都快看不下去了。|带着满足和享受，发出满足惬意的嗯声，温暖舒适。|嗯，这个真好吃，嗯，太满足了。|Guarda che cosa rivoltante, non riesco nemmeno a guardarla."
)
build(){ local n="$1" ev="$2" evt="$3" op="$4" opt="$5"
  QWEN_ACT_MAP=$O/_${n}_event.qamp ./qwen_tts -d $M -s vivian -l Chinese --seed $SEED -T 1.1 -I "$ev" --text "$evt" -o $O/0_SRC_${n}_event.wav >/dev/null 2>&1
  QWEN_ACT_MAP=$O/_${n}_opp.qamp   ./qwen_tts -d $M -s vivian -l Chinese --seed $SEED -T 1.1 -I "$op" --text "$opt" -o $O/0_SRC_${n}_opp.wav   >/dev/null 2>&1
  python3 tests/act_map_steer.py $O/_${n}_opp.qamp $O/_${n}_event.qamp $O/${n}.qlsteer --unit-per-layer >/dev/null 2>&1
  echo "  built ${n}.qlsteer (src event $(dur $O/0_SRC_${n}_event.wav) / opp $(dur $O/0_SRC_${n}_opp.wav))"
}
gen(){ local sp="$1" n="$2" txt="$3" ins="$4" w; w=$(wt $sp)
  ./qwen_tts -d $M $(vf $sp) -l Italian --seed $SEED -T 1.1 --expr $IT --expr-weight $(ew $sp) -I "$ins" \
    --ml-steer $O/${n}.qlsteer --ml-weight $w --ml-range 21-25 --text "$txt" -o $O/${sp}_${n}_w${w}.wav >/dev/null 2>&1
  echo "    ${sp}_${n}_w${w} -> $(dur $O/${sp}_${n}_w${w}.wav)"
}
echo "===== PARA-VEC from CN source -> $O ====="
for e in "${EVENTS[@]}"; do
  IFS='|' read -r n ev evt op opt carrier <<< "$e"
  echo "--- [$n] ---"; build "$n" "$ev" "$evt" "$op" "$opt"
  for sp in galatea vivian ryan; do gen "$sp" "$n" "$carrier" "$ev"; done
done
echo "DONE -> $O"
