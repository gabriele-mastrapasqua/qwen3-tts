#!/usr/bin/env bash
# ============================================================================================
# CSP-FT ITALIAN for the 0.6B model — a SEPARATE, ISOLATED twin of dgx_csp_italian.sh (the 1.7B run).
# NOTHING here touches the 1.7B run: its own RUN_DIR (runs/csp_06b_italian), its own markers/log/outputs,
# its own .expr name (italian_csp_06b.expr). The 1.7B run dir, its checkpoint and presets/expr/*topk6* are
# never read or written by this script.
#
# WHY a native 0.6B FT (analysis 2026-06-30, memory project_06b_emotion_analysis):
#   The 0.6B HAS emotion (~20% activation shift) but ENTANGLED with language (half the width = less
#   separable; emotion identity sits early/mid, on top of pronunciation). So the 1.7B steering-vector win
#   does NOT transfer (linear map gate failed, R²<0) and steering at L21-25 only moves timbre. A native FT
#   can DISENTANGLE in-weights what steering can't reach. The CSP method PROBES which layers carry emotion
#   ON THE 0.6B ITSELF (csp_probe.py reads hidden=1024 from the model) and trains only those (freeze the
#   rest incl. pronunciation) — so we do NOT hardcode the 1.7B's L16-26; the 0.6B picks its own band.
#
#   0.6B ignores --instruct at inference, so the emotion must be FT-CARRIED (.expr) — exactly this output.
#
# Same data as the 1.7B run (EMOVO + Emozionalmente, Italian) — codec-encoded fresh into THIS run dir
# (set REUSE_CODES=/path/to/italian_emotion/train_with_codes.jsonl to skip prep/encode/concat and reuse an
# already-encoded set from the 1.7B run — codec codes are model-agnostic).
#
# Usage (on the DGX, from ~/qwen-ft):
#   SMART=1 nohup bash dgx_csp_06b_italian.sh >/dev/null 2>&1 &
#   tail -f ~/qwen-ft/runs/csp_06b_italian/csp_06b_italian.log
# Override the model dir if it's named differently:  MODEL=/root/qwen-ft/models/0.6B-CustomVoice ...
# ============================================================================================
set -uo pipefail

ROOT="$HOME/qwen-ft"
RUN_DIR="${RUN_DIR:-$ROOT/runs/csp_06b_italian}"   # ISOLATED 0.6B run dir (separate from csp_italian/)
cd "$ROOT"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/csp_06b_italian.log"
TRAIN_IMG="qwen-ft:latest"
TOK="/root/qwen-ft/models/tokenizer-12hz"          # shared, read-only
MODEL="${MODEL:-/root/qwen-ft/models/0.6B-CustomVoice}"   # the 0.6B CV model (override if named differently)
PROBE_JSON="$RUN_DIR/csp_layers_06b_italian.json"
OUT_CKPT="$RUN_DIR/out_csp_06b_italian"
EMOZ_DIR="$RUN_DIR/emozionalmente_zenodo"
EXPR_OUT="$RUN_DIR/italian_csp_06b.expr"
EPOCHS_FT="${EPOCHS_FT:-10}"
TOPK="${TOPK:-6}"                                  # 1.7B win was top_k=6; 0.6B is entangled -> give the FT layers
MINAGREE="${MINAGREE:-0}"

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
done_marker() { echo "$RUN_DIR/$1.DONE"; }
is_done() { [ -f "$(done_marker "$1")" ]; }
mark()    { touch "$(done_marker "$1")"; say "<<< stage '$1' DONE"; }
stage()   { say ">>> stage '$1' START"; }
need_file() { [ -s "$2" ] || { say "FAIL stage '$1': missing/empty output $2"; exit 1; }; }
run_train() { docker run --rm --gpus all --ipc=host -e PYTHONUNBUFFERED=1 \
                -v "$ROOT:/root/qwen-ft" -v "$ROOT:$ROOT" "$TRAIN_IMG" bash -c "$1"; }

say "================ CSP-FT 0.6B ITALIAN START ================"
say "run_dir=$RUN_DIR  model=$MODEL  out=$OUT_CKPT  epochs_ft=$EPOCHS_FT  top_k=$TOPK"

# Preflight: the 0.6B model must exist (clear error instead of a docker failure deep in the run).
if [ ! -f "$MODEL/model.safetensors" ] && [ ! -f "${MODEL/\/root\/qwen-ft/$ROOT}/model.safetensors" ]; then
  say "FAIL: 0.6B model not found at $MODEL"
  say "  models/ on this box:"; ls -1 "$ROOT/models" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
  say "  set MODEL=/root/qwen-ft/models/<the-0.6B-CV-dir> and re-run."
  exit 1
fi

CODES="$RUN_DIR/italian_emotion/train_with_codes.jsonl"
if [ -n "${REUSE_CODES:-}" ]; then
  # Reuse an already-encoded Italian emotion set (codec codes are model-agnostic) -> skip prep/encode/concat.
  if is_done reuse_codes; then say "skip reuse_codes (marker present)"; else
    stage reuse_codes
    mkdir -p "$RUN_DIR/italian_emotion"
    [ -s "$REUSE_CODES" ] || { say "FAIL: REUSE_CODES=$REUSE_CODES missing/empty"; exit 1; }
    cp "$REUSE_CODES" "$CODES"
    need_file reuse_codes "$CODES"; mark reuse_codes
  fi
else
  # ----- STAGE 1: EMOVO prep (host) -----
  if is_done emovo_prep; then say "skip emovo_prep"; else
    stage emovo_prep; mkdir -p "$RUN_DIR/emovo"
    if   [ -s "$RUN_DIR/emovo/train_raw.jsonl" ]; then say "reuse RUN_DIR emovo manifest"
    elif [ -s "$ROOT/emovo/train_raw.jsonl" ]; then cp "$ROOT/emovo/train_raw.jsonl" "$RUN_DIR/emovo/train_raw.jsonl"
    else python3 dgx_emovo_prep.py --out "$RUN_DIR/emovo/train_raw.jsonl" 2>&1 | tee -a "$LOG"; fi
    need_file emovo_prep "$RUN_DIR/emovo/train_raw.jsonl"; mark emovo_prep
  fi
  # ----- STAGE 2: Emozionalmente download + prep (host) -----
  if is_done emoz_prep; then say "skip emoz_prep"; else
    stage emoz_prep
    python3 -c 'import soundfile' 2>/dev/null || pip install --break-system-packages -q soundfile 2>&1 | tail -2
    if [ ! -f "$EMOZ_DIR/emozionalmente/metadata/samples.csv" ]; then
      mkdir -p "$EMOZ_DIR"; say "downloading emozionalmente.zip (~559 MB) ..."
      curl -L -s -o "$EMOZ_DIR/emozionalmente.zip" \
        "https://zenodo.org/records/12616095/files/emozionalmente.zip?download=1"
      unzip -q -o "$EMOZ_DIR/emozionalmente.zip" -d "$EMOZ_DIR" -x "__MACOSX/*"
    fi
    AGREE_FLAG=""
    if [ "${SMART:-0}" = "1" ]; then AGREE_FLAG="--smart-agreement"
    elif [ "$MINAGREE" -gt 0 ]; then AGREE_FLAG="--min-agreement $MINAGREE"; fi
    python3 prepare_emozionalmente.py --local-dir "$EMOZ_DIR" \
        --out "$RUN_DIR/emozionalmente/train_raw.jsonl" --loudnorm $AGREE_FLAG 2>&1 | tee -a "$LOG"
    need_file emoz_prep "$RUN_DIR/emozionalmente/train_raw.jsonl"; mark emoz_prep
  fi
  # ----- STAGE 3: codec-encode (docker) -----
  if is_done encode; then say "skip encode"; else
    stage encode
    run_train "
      cd /root/qwen-ft/Qwen3-TTS/finetuning
      for SET in emovo emozionalmente; do
        python3 -u prepare_data.py --device cuda:0 --tokenizer_model_path $TOK \
          --input_jsonl  $RUN_DIR/\$SET/train_raw.jsonl \
          --output_jsonl $RUN_DIR/\$SET/train_with_codes.jsonl
      done
    " 2>&1 | tee -a "$LOG"
    need_file encode "$RUN_DIR/emovo/train_with_codes.jsonl"
    need_file encode "$RUN_DIR/emozionalmente/train_with_codes.jsonl"; mark encode
  fi
  # ----- STAGE 4: concat -> single Italian set (host) -----
  if is_done concat; then say "skip concat"; else
    stage concat
    python3 concat_manifests.py --out "$CODES" --langs Italian,Italian \
        "$RUN_DIR/emovo/train_with_codes.jsonl" \
        "$RUN_DIR/emozionalmente/train_with_codes.jsonl" 2>&1 | tee -a "$LOG"
    need_file concat "$CODES"; mark concat
  fi
fi

# ----- STAGE 5: CSP probe ON THE 0.6B -> its native emotion layers (docker) -----
if is_done probe; then say "skip probe"; else
  stage probe
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u csp_probe.py --train_jsonl $CODES \
      --init_model_path $MODEL --out_json $PROBE_JSON --epochs 3 --top_k $TOPK
  " 2>&1 | tee -a "$LOG"
  need_file probe "$PROBE_JSON"; mark probe
fi
CSP_LAYERS=$(python3 -c "import json;print(','.join(map(str,json.load(open('$PROBE_JSON'))['selected']['top_k'])))")
say "0.6B probe selected CSP blocks: $CSP_LAYERS  (these are the 0.6B-native emotion layers — expect them to"
say "differ from the 1.7B's; our act-map analysis predicts earlier/more-spread layers on the small model)"
[ -n "$CSP_LAYERS" ] || { say "FAIL: empty CSP layer selection"; exit 1; }

# ----- STAGE 6: CSP-FT on the 0.6B (docker) -----
if is_done train; then say "skip train"; else
  stage train
  run_train "
    cd /root/qwen-ft/Qwen3-TTS/finetuning &&
    python3 -u dgx_sft_expr_csp.py --train_jsonl $CODES \
      --init_model_path $MODEL --output_model_path $OUT_CKPT \
      --csp-layers '$CSP_LAYERS' --scope full --num_epochs $EPOCHS_FT &&
    chmod -R a+rX $OUT_CKPT
  " 2>&1 | tee -a "$LOG"
  need_file train "$OUT_CKPT/checkpoint-final/model.safetensors"; mark train
fi

# ----- STAGE 7: export -> italian_csp_06b.expr (host; expr_extract is pure numpy, --hidden 1024) -----
if is_done export; then say "skip export"; else
  stage export
  python3 -c 'import numpy' 2>/dev/null || pip install --break-system-packages -q numpy 2>&1 | tail -2
  python3 "$ROOT/tests/expr_extract.py" "$MODEL" "$OUT_CKPT/checkpoint-final" "$EXPR_OUT" \
      --lang Italian --hidden 1024 2>&1 | tee -a "$LOG"
  need_file export "$EXPR_OUT"; mark export
fi

say "DONE. 0.6B emotion .expr -> $EXPR_OUT  (target_hidden_size=1024)"
say "  pull to Mac:  scp dgx:$EXPR_OUT presets/expr/italian_csp_06b.expr   (NEW name — never clobbers *topk6)"
say "  try it:       ./qwen_tts -d qwen3-tts-0.6b -s ryan -l Italian -T 1.1 --expr presets/expr/italian_csp_06b.expr --expr-weight 1.0 --text '...'"
say "  CSP blocks:   $CSP_LAYERS   probe: $PROBE_JSON"
say "================ CSP-FT 0.6B ITALIAN ALL DONE ================"
touch "$RUN_DIR/csp_06b_italian.ALLDONE"
