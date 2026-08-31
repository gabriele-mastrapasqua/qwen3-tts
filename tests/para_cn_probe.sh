#!/usr/bin/env bash
set -uo pipefail
cd /Users/gabrielemastrapasqua/source/personal/qwen-tts
O=samples/para_cn_probe; rm -rf $O; mkdir -p $O
M=qwen3-tts-1.7b; SEED=42
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local n="$1" ins="$2" txt="$3"
  ./qwen_tts -d $M -s vivian -l Chinese --seed $SEED -T 1.1 -I "$ins" --text "$txt" -o $O/$n.wav >/dev/null 2>&1
  echo "  $n -> $(dur $O/$n.wav)"
}
gen disgust "用强烈厌恶、恶心反胃的语气说话，在词语之间发出不由自主的干呕和嫌弃的声音。" "呸，这个东西太恶心了，呕，我都快看不下去了。"
gen growl   "用愤怒咆哮的语气，从胸腔发出低沉的怒吼，咬牙切齿，怒不可遏。" "你竟然敢这样对我，哼，我绝对不能接受。"
gen gasp    "突然受惊倒吸一口气，震惊地喘息，难以置信。" "啊，我简直不敢相信，哎呀，太意外了！"
gen laugh   "开怀大笑，咯咯地笑个不停，充满快乐。" "哈哈哈，这真是太好笑了，哈哈，我笑死了！"
gen sigh    "深深地、疲惫地叹一口气，无奈又沮丧。" "唉，今天真是累坏了，唉，我撑不下去了。"
echo "DONE -> $O"
