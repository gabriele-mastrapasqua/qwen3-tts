#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
OUT=samples/icl_emotion
mkdir -p "$OUT"
MODEL=qwen3-tts-1.7b
VOICE=voices/galatea_icl.qvoice
TEXT="Non riesco proprio a crederci. Come hai potuto farmi questo?"
SEED=42
TEMP=1.1
ANGRY="Speak with intense, furious anger, almost shouting, sharp and aggressive."
SAD="Speak with deep sadness and sorrow, your voice trembling, slow and heavy."

gen() { # $1=cap $2=tag $3=instruct(optional)
  local cap=$1 tag=$2 instr=$3
  local caparg=""; [ "$cap" != "full" ] && caparg="--icl-frames $cap"
  local iarg=();   [ -n "$instr" ] && iarg=(--instruct "$instr")
  ./qwen_tts -d "$MODEL" --load-voice "$VOICE" -l Italian --seed "$SEED" -T "$TEMP" \
    $caparg "${iarg[@]}" --text "$TEXT" -o "$OUT/${tag}_${cap}.wav" --silent 2>/dev/null
  echo "  done ${tag}_${cap}.wav"
}

for cap in full 200 120 80 50 30; do
  echo "== cap=$cap =="
  gen "$cap" neu ""
  gen "$cap" ang "$ANGRY"
  gen "$cap" sad "$SAD"
done
echo "ALL DONE -> $OUT"
