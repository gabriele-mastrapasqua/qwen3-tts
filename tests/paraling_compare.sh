#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
O=samples/paraling_compare; mkdir -p $O
FT="${FT:-presets/expr/paraling_csp_no0_mid_ep5.expr}"   # override with FT=... to compare variants
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
LAUGH="Che bella notizia, davvero, [laugh] non ci posso credere."
SIGH="Sono davvero stanco oggi, [sigh] non ce la faccio piu."
COUGH="Scusami un attimo, [cough] dicevo che dobbiamo andare."

voiceflags(){ case "$1" in
  ryan)    echo "-s ryan";;
  vivian)  echo "-s vivian";;
  galatea) echo "--load-voice voices/galatea_graft.qvoice --icl-only";;
esac; }

m1(){ local v="$1" mk="$2" txt="$3"; ./qwen_tts -d qwen3-tts-1.7b $(voiceflags $v) -l Italian --seed 42 -T 1.1 --text "$txt" -o $O/${v}_${mk}_M1macro.wav >/dev/null 2>&1; echo "  ${v}_${mk}_M1macro -> $(dur $O/${v}_${mk}_M1macro.wav)"; }
m2(){ local v="$1" mk="$2" txt="$3"; ./qwen_tts -d qwen3-tts-1.7b $(voiceflags $v) -l Italian --seed 42 -T 1.1 --no-compose --expr "$FT" --expr-weight 1.0 --text "$txt" -o $O/${v}_${mk}_M2ft.wav >/dev/null 2>&1; echo "  ${v}_${mk}_M2ft -> $(dur $O/${v}_${mk}_M2ft.wav)"; }

echo "FT variant = $FT"
for v in ryan vivian galatea; do
  echo "===== $v ====="
  m1 $v laugh "$LAUGH"; m2 $v laugh "$LAUGH"
  m1 $v sigh  "$SIGH";  m2 $v sigh  "$SIGH"
  m2 $v cough "$COUGH"
done
echo "===== DONE -> $O ====="
