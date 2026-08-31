#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; MODEL=qwen3-tts-1.7b; SEED=42; T=1.1
IT=presets/expr/italian_csp_topk6.expr
D=samples/emo_retest_0622
GV=voices/galatea_graft.qvoice
CAP_T="Oggi ho camminato fino al mercato e ho comprato del pane e del latte."
NEU_I="Speak in a plain, neutral, matter-of-fact tone."
ANG_I="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
SAD_I="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
ANG_T="Come ti permetti di parlarmi così? Questo non lo accetto, è inaccettabile!"
SAD_T="Ho perso tutto quello che avevo, e adesso non so più cosa fare."
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local out="$1"; shift; $BIN -d $MODEL -l Italian --seed $SEED -T $T "$@" -o "$out" >/dev/null 2>&1; echo "  $(basename "$out") -> $(dur "$out")"; }
capft(){ # capft <voiceflags...> -- <outqamp> <instruct>
  local vf=(); while [ "$1" != "--" ]; do vf+=("$1"); shift; done; shift
  local out="$1"; local ins="$2"
  QWEN_ACT_MAP="$out" $BIN -d $MODEL -l Italian --seed $SEED -T $T "${vf[@]}" --expr $IT --expr-weight 1.2 -I "$ins" --text "$CAP_T" -o /dev/null >/dev/null 2>&1
  echo "  captured $(basename "$out")"
}

echo "===== IDEA 2: FT-space / voice-native steer capture ====="
capft -s vivian -- $D/vivian_ft_neutral.qamp "$NEU_I"
capft -s vivian -- $D/vivian_ft_angcap.qamp  "$ANG_I"
capft -s vivian -- $D/vivian_ft_sadcap.qamp  "$SAD_I"
python3 tests/act_map_steer.py $D/vivian_ft_neutral.qamp $D/vivian_ft_angcap.qamp $D/vivian_ang_ft.qlsteer --unit-per-layer >/dev/null
python3 tests/act_map_steer.py $D/vivian_ft_neutral.qamp $D/vivian_ft_sadcap.qamp $D/vivian_sad_ft.qlsteer --unit-per-layer >/dev/null
capft --load-voice $GV --icl-only -- $D/galatea_ft_neutral.qamp "$NEU_I"
capft --load-voice $GV --icl-only -- $D/galatea_ft_angcap.qamp  "$ANG_I"
capft --load-voice $GV --icl-only -- $D/galatea_ft_sadcap.qamp  "$SAD_I"
python3 tests/act_map_steer.py $D/galatea_ft_neutral.qamp $D/galatea_ft_angcap.qamp $D/galatea_ang_ft.qlsteer --unit-per-layer >/dev/null
python3 tests/act_map_steer.py $D/galatea_ft_neutral.qamp $D/galatea_ft_sadcap.qamp $D/galatea_sad_ft.qlsteer --unit-per-layer >/dev/null

O=samples/idea2_ftspace; mkdir -p $O
echo "--- demo: combine with RYAN-captured (baseline) vs VOICE-NATIVE direction ---"
gen $O/vivian_anger_ryanDir.wav  -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_ang.qlsteer     --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/vivian_anger_nativeDir.wav -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/vivian_ang_ft.qlsteer --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/vivian_sad_ryanDir.wav    -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_sad.qlsteer     --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/vivian_sad_nativeDir.wav   -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/vivian_sad_ft.qlsteer --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/galatea_anger_ryanDir.wav   --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_ang.qlsteer      --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/galatea_anger_nativeDir.wav --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/galatea_ang_ft.qlsteer --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/galatea_sad_ryanDir.wav     --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_sad.qlsteer      --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/galatea_sad_nativeDir.wav   --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/galatea_sad_ft.qlsteer --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"

echo "===== IDEA 3: cleaned direction (mean-center + project-out energy) vs original ====="
O=samples/idea3_clean; mkdir -p $O
gen $O/vivian_anger_orig.wav  -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_ang.qlsteer       --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/vivian_anger_clean.wav -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_ang_clean.qlsteer --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/vivian_sad_orig.wav    -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_sad.qlsteer       --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/vivian_sad_clean.wav   -s vivian --expr $IT --expr-weight 1.2 --ml-steer $D/ryan_sad_clean.qlsteer --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/galatea_anger_orig.wav  --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_ang.qlsteer       --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/galatea_anger_clean.wav --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_ang_clean.qlsteer --ml-weight 8 --ml-range 21-25 -I "$ANG_I" --text "$ANG_T"
gen $O/galatea_sad_orig.wav    --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_sad.qlsteer       --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
gen $O/galatea_sad_clean.wav   --load-voice $GV --icl-only --expr $IT --expr-weight 1.0 --ml-steer $D/ryan_sad_clean.qlsteer --ml-weight 8 --ml-range 21-25 -I "$SAD_I" --text "$SAD_T"
echo "===== idea2+3 DONE ====="
