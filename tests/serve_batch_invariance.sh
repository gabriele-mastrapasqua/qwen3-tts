#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-qwen3-tts-0.6b}"
SPK="${SPK:-ryan}"
LANG_="${LANG_:-english}"
QUANT="${QUANT:---int8}"
BATCH="${BATCH:-4}"
J="${J:-4}"
PORT="${PORT:-8953}"
EP="${EP:-/v1/tts/stream}"
OUT="${OUT:-/tmp/tts/batch_invariance}"
BIN=./qwen_tts

R_TEXT="${R_TEXT:-The engineer will reach your side before two in the afternoon, and if he does not show up I can book another one for you.}"
R_SEED="${R_SEED:-4242}"
LONG_TEXT="Please hold the line while I check the account, the payment has not been posted yet but it will show before the evening, and if it still does not appear by Friday I will escalate it for you and call you back."
SHORT_TEXT="One moment please."

[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
[ -d "$MODEL" ] || { echo "🚨 modello assente: $MODEL"; exit 1; }
if curl -s -m 2 "http://localhost:$PORT/v1/health" 2>/dev/null | grep -q '"ok"'; then
    echo "🚨 something already answers on port $PORT: stop it, or the gate is void."; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"

req() {  # $1 = file di uscita, $2 = testo, $3 = seed
    local body="$1.body.json"
    python3 - "$2" "$SPK" "$LANG_" "$3" > "$body" <<'PY'
import json, sys
print(json.dumps({"text": sys.argv[1], "speaker": sys.argv[2], "language": sys.argv[3],
                  "seed": int(sys.argv[4]), "temperature": 0.0}))
PY
    curl -s -m 300 -X POST "http://localhost:$PORT$EP" \
         -H 'content-type: application/json' --data-binary @"$body" -o "$1"
    rm -f "$body"
}
dur() { python3 -c "import os,sys; print('%.2f' % (os.path.getsize(sys.argv[1])/2/24000))" "$1"; }
sum_() { md5sum "$1" 2>/dev/null | cut -c1-12 || md5 -q "$1" | cut -c1-12; }

echo "=== batch invariance: the same request, different neighbours ==="
$BIN --caps 2>&1 | grep -E "^  (build|flag attive|  )" | sed "s/^/  /"
echo "  $MODEL $QUANT · $SPK · -j$J --batch-size $BATCH"
echo "  endpoint $EP · pin: ${PIN_ENV:-nessuno (percorsi veri)} · greedy · seed $R_SEED"
echo "  criterio: ${CRITERION:-byte}"
echo

PIN_ENV="${PIN_ENV:-}"
$PIN_ENV $BIN -d "$MODEL" $QUANT --serve "$PORT" --batch-size "$BATCH" -j "$J" \
    > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill -9 $SRV 2>/dev/null' EXIT
for _ in $(seq 1 240); do
    curl -s -m 2 "http://localhost:$PORT/v1/health" 2>/dev/null | grep -q '"ok"' && break
    sleep 1
done
curl -s -m 2 "http://localhost:$PORT/v1/health" | grep -q '"ok"' || {
    echo "🚨 server non partito"; tail -20 "$OUT/server.log"; exit 1; }
req "$OUT/_warm.pcm" "Warm up." 7      # the first generation pays page faults, not the machine

req "$OUT/ref.pcm" "$R_TEXT" "$R_SEED"
echo "  riferimento (da sola)              $(dur "$OUT/ref.pcm") s   $(sum_ "$OUT/ref.pcm")"

req "$OUT/_short.pcm" "$SHORT_TEXT" 11 &
F1=$!
sleep 0.2
req "$OUT/a.pcm" "$R_TEXT" "$R_SEED"
wait $F1
echo "  A · resta sola (N->1)              $(dur "$OUT/a.pcm") s   $(sum_ "$OUT/a.pcm")"

req "$OUT/_long.pcm" "$LONG_TEXT" 12 &
F2=$!
sleep 3
req "$OUT/b.pcm" "$R_TEXT" "$R_SEED"
wait $F2
echo "  B · ammessa accanto (1->N)         $(dur "$OUT/b.pcm") s   $(sum_ "$OUT/b.pcm")"

req "$OUT/_long2.pcm" "$LONG_TEXT" 13 &
F3=$!
sleep 3
req "$OUT/_short2.pcm" "$SHORT_TEXT" 14 &
F4=$!
sleep 0.2
req "$OUT/c.pcm" "$R_TEXT" "$R_SEED"
wait $F3 $F4
echo "  C · ammessa accanto e poi sola     $(dur "$OUT/c.pcm") s   $(sum_ "$OUT/c.pcm")"

echo
frames() { python3 -c "import os,sys; print(os.path.getsize(sys.argv[1])//2)" "$1"; }
rc=0
for arm in a b c; do
    if [ "${CRITERION:-byte}" = byte ]; then
        if cmp -s "$OUT/ref.pcm" "$OUT/$arm.pcm"; then
            echo "  ✅ $arm identica al riferimento (byte a byte)"
        else
            echo "  ❌ $arm DIVERSA dal riferimento: $(dur "$OUT/ref.pcm") s contro $(dur "$OUT/$arm.pcm") s"
            rc=1
        fi
    else
        fr=$(frames "$OUT/ref.pcm"); fa=$(frames "$OUT/$arm.pcm")
        if [ "$fr" = "$fa" ]; then
            if cmp -s "$OUT/ref.pcm" "$OUT/$arm.pcm"; then
                echo "  ✅ $arm same duration ($fr samples) and identical bytes"
            else
                echo "  ✅ $arm same duration ($fr samples); different bytes = summation order, expected"
            fi
        else
            echo "  ❌ $arm DIFFERENT DURATION: $fr samples vs $fa ($(dur "$OUT/ref.pcm") s vs $(dur "$OUT/$arm.pcm") s)"
            rc=1
        fi
    fi
done
echo
if [ "$rc" = 0 ]; then
    echo "batch-invariance [${CRITERION:-byte}]: PASS — la composizione del batch non cambia l'uscita."
else
    echo "batch-invariance [${CRITERION:-byte}]: FAIL"
    echo "  Look first at bb->act_idx / bb->B_eff: they must describe THIS frame's slots on"
    echo "  every exit path (qwen_batch_pack_active), and must be written BEFORE the codec"
    echo "  head, not inherited from the Talker step at the end of the loop."
    echo "  pcm grezzi in $OUT"
fi
exit $rc
