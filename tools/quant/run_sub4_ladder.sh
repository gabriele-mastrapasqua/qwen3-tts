#!/bin/bash
# run_sub4_ladder.sh — E7.2 sub-4-bit quality gate (see docs/quant-sub4.md §3).
# Phase A: bf16 rails. Phase B: teacher-forced replay per variant (C int8/int4 refs +
# Python fake-quant formats). Output: per-codebook agreement via tests/quant_ladder.py.
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3-tts-0.6b}
OUT=${OUT:-samples/tests/2026-07-14_quant-sub4-ladder}
SCRATCH=${SCRATCH:-/tmp/quant-sub4}
TEXT="La prova di oggi misura quanto il quantizzatore riesce a conservare la voce. Se anche i dettagli più fini restano stabili, il formato regge e possiamo dimezzare la memoria del modello."
ARGS=(--seed 42 -s ryan -l Italian --temperature 0 -j1 --silent)

mkdir -p "$OUT" "$SCRATCH"

run_tf () { # $1=model-dir $2=label $3=cp-prec (bf16|int8|int4)
    rm -f "$OUT/$2.codes"
    env QWEN_CP_PREC="$3" QWEN_TF_CODES="$OUT/bf16.codes" QWEN_DUMP_CODES="$OUT/$2.codes" \
        ./qwen_tts -d "$1" --text "$TEXT" "${ARGS[@]}" -o "$SCRATCH/$2.wav"
    echo "== $2: $(wc -l < "$OUT/$2.codes") frames"
}

echo "=== Phase A: bf16 rails ==="
rm -f "$OUT/bf16.codes"
QWEN_DUMP_CODES="$OUT/bf16.codes" \
    ./qwen_tts -d "$MODEL" --text "$TEXT" "${ARGS[@]}" -o "$OUT/rails.wav"
echo "== rails: $(wc -l < "$OUT/bf16.codes") frames"

echo "=== Phase B: teacher-forced replays ==="
run_tf "$MODEL" bf16_tf bf16                  # control: must be 100%
run_tf "$MODEL" int8_c  int8                  # C int8 reference (June: 78%)
run_tf "$MODEL" int4_c  int4                  # C int4 reference (June: 46%)

for fmt in q4_0 q3_0 q3_k q2_k q2_0; do
    echo "--- fake-quant $fmt ---"
    rm -rf "$SCRATCH/mv_$fmt"
    python3 tools/quant/fakequant_cp.py --model "$MODEL" --out "$SCRATCH/mv_$fmt" \
        --format "$fmt" --scope all --weights x2 | tail -3
    run_tf "$SCRATCH/mv_$fmt" "$fmt" bf16     # quant error is baked into the weights
    rm -rf "$SCRATCH/mv_$fmt"                 # keep disk flat (1.8GB each)
done

echo "=== Ladder ==="
python3 tests/quant_ladder.py \
    bf16:"$OUT/bf16.codes" bf16_tf:"$OUT/bf16_tf.codes" \
    int8:"$OUT/int8_c.codes" int4:"$OUT/int4_c.codes" \
    q4_0:"$OUT/q4_0.codes" q3_0:"$OUT/q3_0.codes" \
    q3_k:"$OUT/q3_k.codes" q2_k:"$OUT/q2_k.codes" q2_0:"$OUT/q2_0.codes" \
    | tee "$OUT/ladder_results.txt"
