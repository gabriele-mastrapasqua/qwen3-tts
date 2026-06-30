#!/usr/bin/env bash
# Sync the 0.6B CSP-FT run to the DGX (separate from the 1.7B sync). Idempotent (scp overwrites).
#   bash training/expressivity-lora/sync_06b_to_dgx.sh           # host alias `dgx`, ~/qwen-ft
#   DGX=myhost ROOT=/data/qwen-ft bash training/expressivity-lora/sync_06b_to_dgx.sh
#
# Lands files where dgx_csp_06b_italian.sh expects them:
#   ~/qwen-ft/                          the 0.6B orchestrator + host-run prep scripts
#   ~/qwen-ft/tests/                    expr_extract.py (the .expr export, --hidden 1024)
#   ~/qwen-ft/Qwen3-TTS/finetuning/     docker-run trainers (shared, model-agnostic)
set -euo pipefail
DGX="${DGX:-dgx}"
ROOT="${ROOT:-qwen-ft}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

echo "[sync-06b] target: $DGX:$ROOT"
ssh "$DGX" "mkdir -p $ROOT/tests $ROOT/Qwen3-TTS/finetuning"

echo "[sync-06b] orchestrator + host prep -> $ROOT/"
scp "$HERE/dgx_csp_06b_italian.sh" "$HERE/prepare_emozionalmente.py" "$HERE/concat_manifests.py" \
    "$HERE/dgx_emovo_prep.py" "$DGX:$ROOT/"

echo "[sync-06b] export tool -> $ROOT/tests/"
scp "$REPO/tests/expr_extract.py" "$DGX:$ROOT/tests/"

echo "[sync-06b] docker-run trainers (shared, model-agnostic) -> $ROOT/Qwen3-TTS/finetuning/"
scp "$HERE/csp_probe.py" "$HERE/dgx_sft_expr_csp.py" "$HERE/dgx_dataset_expr_lang.py" \
    "$HERE/prepare_data.py" "$DGX:$ROOT/Qwen3-TTS/finetuning/"

echo "[sync-06b] done. On the DGX:"
echo "    SMART=1 nohup bash dgx_csp_06b_italian.sh >/dev/null 2>&1 &"
echo "    tail -f ~/qwen-ft/runs/csp_06b_italian/csp_06b_italian.log"
echo "  If your 1.7B run already encoded the Italian set, skip prep/encode by reusing its codes:"
echo "    REUSE_CODES=~/qwen-ft/runs/csp_italian/italian_emotion/train_with_codes.jsonl SMART=1 \\"
echo "      nohup bash dgx_csp_06b_italian.sh >/dev/null 2>&1 &"
