#!/usr/bin/env bash
# ============================================================================
# PARA-FT STEP 1 — GATE EVAL
#
# Runs Mac-side, NO GPU. The instant Step 1 training finishes on the GPU box:
#   1. pull the chosen LoRA adapter (16MB) from the GPU box
#   2. export_expr.py -> factored .expr (no base needed, pure numpy)
#   3. synth [laugh]/[sigh] phrases (EN + IT) WITH vs WITHOUT the .expr (A/B)
#
# GATE PASS = the [tag] becomes a REAL event (laugh/sigh sound) in the data's
#   language, with NO truncation / NO garble, and the no-tag control stays clean.
# If pass -> LR was the bug, confirmed. Proceed to Step 2 (clean EN data).
#
# Usage:
#   tests/para_gate_eval.sh [EPOCH] [WEIGHT]
#     EPOCH  : adapter-epN to test (default: final)   e.g. 8
#     WEIGHT : --expr-weight dose (default: 1.0)       try 0.7 1.0 1.3
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${GPU:-gpubox}"
REMOTE_OUT="/home/gabriele/qwen-ft/out_para_step1_lr2e6_s031"
EP="${1:-final}"
W="${2:-1.0}"
ADIR="adapter-${EP}"
MODEL="qwen3-tts-1.7b"
OUTD="samples/para_gate_step1"
LOCAL_ADAPTER="/tmp/para_step1_${ADIR}"
EXPR="/tmp/para_step1_${EP}.expr"
mkdir -p "$OUTD"

echo "### [1/3] pull adapter $ADIR from $GPU"
rm -rf "$LOCAL_ADAPTER"; mkdir -p "$LOCAL_ADAPTER"
scp "$GPU:$REMOTE_OUT/$ADIR/adapter_model.safetensors" "$GPU:$REMOTE_OUT/$ADIR/adapter_config.json" "$LOCAL_ADAPTER/" || { echo "!! adapter $ADIR not found on GPU box"; exit 1; }

echo "### [2/3] export_expr.py -> $EXPR (factored)"
python3 training/expressivity-lora/export_expr.py "$LOCAL_ADAPTER" "$EXPR" --lang English --hidden 2048 || exit 1
ls -la "$EXPR"

echo "### [3/3] synth A/B (baseline vs +expr w=$W), seed 42 ryan 1.7b"
# tag phrases (markers trained = [laugh],[sigh]) + a no-tag control
synth() { # name lang text extra
  local name="$1" lang="$2" text="$3"; shift 3
  ./qwen_tts -d "$MODEL" -s ryan -l "$lang" --seed 42 --text "$text" "$@" -o "$OUTD/$name.wav" --silent 2>/dev/null \
    && echo "   wrote $OUTD/$name.wav"
}
# --- EN ---
synth en_laugh_base    English "That is the funniest thing I have heard all week [laugh] I cannot even."
synth en_laugh_expr    English "That is the funniest thing I have heard all week [laugh] I cannot even." --expr "$EXPR" --expr-weight "$W"
synth en_sigh_base     English "Well [sigh] I suppose we have to start all over again."
synth en_sigh_expr     English "Well [sigh] I suppose we have to start all over again." --expr "$EXPR" --expr-weight "$W"
synth en_ctrl_expr     English "Let me tell you exactly what happened yesterday afternoon." --expr "$EXPR" --expr-weight "$W"
# --- IT (cross-language probe — Step 4 question, informational) ---
synth it_laugh_expr    Italian "È la cosa più divertente che abbia sentito [laugh] non ci posso credere." --expr "$EXPR" --expr-weight "$W"
synth it_sigh_expr     Italian "Va bene [sigh] ricominciamo tutto da capo." --expr "$EXPR" --expr-weight "$W"

echo "### DONE. A/B listen:"
echo "   afplay $OUTD/en_laugh_base.wav ; afplay $OUTD/en_laugh_expr.wav   # [laugh] event?"
echo "   afplay $OUTD/en_sigh_base.wav  ; afplay $OUTD/en_sigh_expr.wav    # [sigh] event?"
echo "   afplay $OUTD/en_ctrl_expr.wav                                     # no-tag stays clean?"
echo "   afplay $OUTD/it_laugh_expr.wav ; afplay $OUTD/it_sigh_expr.wav    # cross-lang (info)"
