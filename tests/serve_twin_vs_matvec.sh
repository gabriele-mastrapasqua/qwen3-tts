#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
if [ -z "${QWEN_BENCH_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export QWEN_BENCH_CAFFEINATED=1; exec caffeinate -i -s "$0" "$@"
fi

MODEL="${MODEL:-qwen3-tts-0.6b}"
QUANT="${QUANT:---int8}"
SPK="${SPK:-ryan}"; LANG_="${LANG_:-english}"
CONC="${CONC:-4}"; REQS="${REQS:-8}"; BATCH="${BATCH:-4}"; J="${J:-4}"
PORT="${PORT:-8951}"; OUT="${OUT:-/tmp/tts/twin_ab}"
BANK="${BANK:-tests/load_texts_en.txt}"; CLASSES="${CLASSES:-short,medium}"
BIN=./qwen_tts
[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"

run() {  # $1 = nome, $2 = 1 se forza i matvec
    local name="$1" force="$2" pid
    if [ "$force" = 1 ]; then export QWEN_BATCH_FORCE_MATVEC=1; else unset QWEN_BATCH_FORCE_MATVEC; fi
    QWEN_BATCH_STATS=1 $BIN -d "$MODEL" $QUANT --serve "$PORT" --batch-size "$BATCH" -j "$J" \
        > "$OUT/${name}.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 180); do curl -s -m 2 "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && break; sleep 1; done
    curl -s -m 300 -o /dev/null -X POST "http://localhost:$PORT/v1/tts" \
        -d "{\"text\":\"Warm up before measuring.\",\"speaker\":\"$SPK\",\"language\":\"$LANG_\",\"seed\":7}"
    python3 tests/load_test.py --url "http://localhost:$PORT" --concurrency "$CONC" \
        --requests "$REQS" --text-file "$BANK" --classes "$CLASSES" --speaker "$SPK" \
        --language "$LANG_" --no-save-audio --json "$OUT/${name}.json" >/dev/null 2>&1
    kill -TERM "$pid" 2>/dev/null; sleep 4
}

echo "model $MODEL ($QUANT) · c=$CONC · $REQS requests · --batch-size $BATCH · -j $J"
echo "  braccio A: dispatcher normale (gemello batched)"
run twin 0
echo "  braccio B: QWEN_BATCH_FORCE_MATVEC=1 (B matvec, bit-esatti)"
run matvec 1

python3 - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
rows = {}
for name in ("twin", "matvec"):
    f = f"{out}/{name}.json"
    if not os.path.exists(f): continue
    d = json.load(open(f)); rows[name] = d[0] if isinstance(d, list) else d
if len(rows) < 2:
    print("un braccio non ha prodotto dati"); sys.exit(1)
print(f"\n{'braccio':<10}{'err':>5}{'Q':>9}{'TTFA p50':>11}{'p95':>9}{'RTF p50':>10}{'p95':>9}")
for n, s in rows.items():
    print(f"{n:<10}{s['errors']:>5}{s['throughput_Q']:>9.2f}{s['ttfa_p50']:>11.0f}"
          f"{s['ttfa_p95']:>9.0f}{s['rtf_p50']:>10.2f}{s['rtf_p95']:>9.2f}")
t, m = rows["twin"], rows["matvec"]
dq = (m["throughput_Q"] / t["throughput_Q"] - 1) * 100 if t["throughput_Q"] else float("nan")
dr = (m["rtf_p50"] / t["rtf_p50"] - 1) * 100 if t["rtf_p50"] else float("nan")
print(f"\nforzando i matvec: throughput {dq:+.1f}%  ·  RTF p50 {dr:+.1f}%")
print("If throughput RISES, the batched twin is losing on this machine and the")
print("dispatcher deve scegliere in base al box, non solo all'ISA. Su un box con VNNI o")
print("On AMX the same question must be asked again: there the answer should invert.")
PY
echo "log dei server (con l'audit dei kernel in coda): $OUT/*.log"
