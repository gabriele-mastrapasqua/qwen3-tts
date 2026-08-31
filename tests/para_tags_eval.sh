#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-gpubox}"
OUTDIR_NAME="${1:-out_para_step2_8tag}"
EP="${2:-final}"
W="${3:-1.5}"
REMOTE_OUT="${REMOTE_HOME:-$HOME}/qwen-ft/$OUTDIR_NAME"
ADIR="adapter-${EP}"
MODEL="qwen3-tts-1.7b"
OUTD="samples/para_tags_${OUTDIR_NAME}"
LOCAL_ADAPTER="/tmp/${OUTDIR_NAME}_${ADIR}"
EXPR="/tmp/${OUTDIR_NAME}_${EP}.expr"
mkdir -p "$OUTD"

echo "### pull $ADIR from $GPU:$OUTDIR_NAME"
rm -rf "$LOCAL_ADAPTER"; mkdir -p "$LOCAL_ADAPTER"
scp "$GPU:$REMOTE_OUT/$ADIR/adapter_model.safetensors" "$GPU:$REMOTE_OUT/$ADIR/adapter_config.json" "$LOCAL_ADAPTER/" || { echo "!! adapter not found"; exit 1; }
echo "### export -> $EXPR"
python3 training/expressivity-lora/export_expr.py "$LOCAL_ADAPTER" "$EXPR" --lang English --hidden 2048 || exit 1

synth() { # name text extra...
  local name="$1" text="$2"; shift 2
  ./qwen_tts -d "$MODEL" -s ryan -l English --seed 42 --no-compose --text "$text" "$@" -o "$OUTD/$name.wav" --silent 2>/dev/null && echo "   wrote $name.wav"
}
echo "### synth all 8 tags @ w=$W (+ baseline + no-tag control)"
synth tag_laugh   "That is the funniest thing I have heard all week [laugh] I cannot even."        --expr "$EXPR" --expr-weight "$W"
synth tag_sigh    "Well [sigh] I suppose we have to start all over again."                          --expr "$EXPR" --expr-weight "$W"
synth tag_cough   "Excuse me for a second [cough] sorry, where was I."                              --expr "$EXPR" --expr-weight "$W"
synth tag_sniff   "It has been a long day [sniff] but we are almost there."                         --expr "$EXPR" --expr-weight "$W"
synth tag_breath  "Okay [breath] let me explain the whole thing from the start."                   --expr "$EXPR" --expr-weight "$W"
synth tag_grunt   "He lifted the box [grunt] and set it down by the door."                          --expr "$EXPR" --expr-weight "$W"
synth tag_sneeze  "The dust everywhere made me [sneeze] excuse me, allergies."                      --expr "$EXPR" --expr-weight "$W"
synth tag_yawn    "It is getting really late [yawn] I should head to bed."                          --expr "$EXPR" --expr-weight "$W"
synth ctrl_notag  "Let me tell you exactly what happened yesterday afternoon."                      --expr "$EXPR" --expr-weight "$W"
synth sigh_base   "Well [sigh] I suppose we have to start all over again."

echo "### DONE -> $OUTD/   (afplay each; KEY: sigh != laugh, control stays clean)"
for w in "$OUTD"/*.wav; do d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$w" 2>/dev/null); printf "   %-16s %ss\n" "$(basename "$w")" "$d"; done
