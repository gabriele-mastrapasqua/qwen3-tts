#!/usr/bin/env bash
cd "$(dirname "$0")/.."
OUT=samples/icl_levers
mkdir -p "$OUT"
M=qwen3-tts-1.7b
GRAFT=voices/galatea_graft.qvoice   # 0 ref_codes -> x-vector graft via --icl-only
ICL=voices/galatea_icl.qvoice     # 375 ref_codes -> faithful
TEXT="Non riesco proprio a crederci. Come hai potuto farmi questo?"
S=42
ANGRY="Speak with intense, furious anger, almost shouting, sharp and aggressive."
SAD="Speak with deep sadness and sorrow, your voice trembling, slow and heavy."

run() { # name + args...
  local name=$1; shift
  echo ">> $name"
  timeout 120 ./qwen_tts -d "$M" --seed "$S" --max-duration 15 -l Italian \
    --text "$TEXT" -o "$OUT/$name.wav" --silent "$@" 2>/dev/null
  local rc=$?
  [ $rc -eq 124 ] && echo "   !! TIMEOUT (runaway) $name"
  local sz=$(stat -f%z "$OUT/$name.wav" 2>/dev/null || echo 0)
  echo "   $name.wav ${sz}B"
}

echo "=== A) GRAFT (yesterday's TOP): x-vector + CV weights + EN instruct, T0.9 ==="
run graft_neu  --load-voice "$GRAFT" --icl-only -T 0.9
run graft_ang  --load-voice "$GRAFT" --icl-only -T 0.9 --instruct "$ANGRY"
run graft_sad  --load-voice "$GRAFT" --icl-only -T 0.9 --instruct "$SAD"

echo "=== B) Dawizzer sampling-knobs on FAITHFUL ICL-lite (no instruct) ==="
run icl_neu    --load-voice "$ICL" -T 0.9
run icl_ang_s1 --load-voice "$ICL" -T 0.9 -p 0.85 -r 1.25
run icl_ang_s2 --load-voice "$ICL" -T 1.1 -p 0.80 -r 1.30
run icl_sad_s1 --load-voice "$ICL" -T 0.7 -p 0.95 -r 1.10
run icl_sad_s2 --load-voice "$ICL" -T 0.6 -p 0.90 -r 1.15
echo "=== C) STACK: ICL + EN-instruct + sampling knobs ==="
run icl_ang_stack --load-voice "$ICL" -T 1.1 -p 0.85 -r 1.25 --instruct "$ANGRY"
run icl_sad_stack --load-voice "$ICL" -T 0.7 -p 0.95 -r 1.10 --instruct "$SAD"
echo "ALL DONE -> $OUT"
