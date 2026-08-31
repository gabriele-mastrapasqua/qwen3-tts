#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
O=samples/paraling_csp; mkdir -p $O
B="-d qwen3-tts-1.7b -s ryan -l Italian --seed 42 -T 1.1"
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
nc(){ ./qwen_tts $B --no-compose --expr "$1" --expr-weight "$2" --text "$3" -o $O/"$4" >/dev/null 2>&1; echo "  $4 -> $(dur $O/$4)"; }
LAUGH="Che bella notizia, davvero, [laugh] non ci posso credere."
SIGH="Sono davvero stanco oggi, [sigh] non ce la faccio piu."
COUGH="Scusami un attimo, [cough] dicevo che dobbiamo andare."

for V in k4_ep5 k4_ep8 k6_ep5; do
  E=presets/expr/paralinguistic_csp_${V}.expr; [ -f "$E" ] || E=presets/expr/paraling_csp_${V}.expr
  echo "=== CSP $V ==="
  nc "$E" 1.0 "$LAUGH" ${V}_laugh.wav
  nc "$E" 1.0 "$SIGH"  ${V}_sigh.wav
  nc "$E" 1.0 "$COUGH" ${V}_cough.wav
done
echo "=== OLD aug (blind L16-26) reference ==="
nc presets/expr/paralinguistic_aug.expr 1.0 "$LAUGH" OLDaug_laugh.wav
nc presets/expr/paralinguistic_aug.expr 1.0 "$SIGH"  OLDaug_sigh.wav
echo "=== method-1 MACRO (auto-compose) ==="
./qwen_tts $B --text "$LAUGH" -o $O/macro_laugh.wav >/dev/null 2>&1; echo "  macro_laugh.wav -> $(dur $O/macro_laugh.wav)"
./qwen_tts $B --text "$SIGH"  -o $O/macro_sigh.wav  >/dev/null 2>&1; echo "  macro_sigh.wav -> $(dur $O/macro_sigh.wav)"
echo "=== DONE ==="
