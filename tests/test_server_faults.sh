#!/bin/sh
# Provoked failures against a live batched server, and the invariants each must keep:
# model work stops for a client that is gone, the request ends exactly once in the
# books, a neighbour's audio does not change, RSS does not grow with the aborts.
#
# The assertions live in tests/fault_probe.py, next to the bytes that provoke them;
# this script only starts the server, runs the probe and reports.
#
# Usage: tests/test_server_faults.sh [model_dir] [port]
#   FAULT_CASES="zombie"   which probe cases to run (default: every case)
#   QWEN_CANCEL_ON_DISCONNECT, and any other QWEN_* in the environment, reach the server.
# Exit: 0 ok, 1 fail, 77 skip (model or binary missing).
MODEL_DIR="${1:-qwen3-tts-0.6b}"
PORT="${2:-8871}"
BIN="${BIN:-./qwen_tts}"
[ -f "$MODEL_DIR/config.json" ] || { echo "server-faults SKIP: no model at $MODEL_DIR"; exit 77; }
[ -x "$BIN" ] || { echo "server-faults SKIP: no binary $BIN"; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/qwen_tts_faults.XXXXXX") || exit 1
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    [ "${FAULT_KEEP:-0}" = 1 ] && echo "server log kept in $TMP" || rm -rf "$TMP"
}
trap cleanup EXIT

BATCH=4
QUEUE=1
echo "server-faults: $BIN -d $MODEL_DIR --serve $PORT --batch-size $BATCH --max-queue $QUEUE" \
     "(QWEN_CANCEL_ON_DISCONNECT=${QWEN_CANCEL_ON_DISCONNECT-unset})"
QWEN_LIFE_TRACE=1 QWEN_REQ_TRACE=1 \
"$BIN" -d "$MODEL_DIR" --serve "$PORT" --batch-size $BATCH --max-queue $QUEUE \
    > "$TMP/srv.log" 2>&1 &
SRV_PID=$!
ready=0
for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/health" 2>/dev/null | grep -q '"ok"'; then ready=1; break; fi
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 0.5
done
[ $ready -eq 1 ] || { echo "server-faults FAIL: server never became ready"; tail -30 "$TMP/srv.log"; exit 1; }
echo "server-faults: server pid $SRV_PID ready, log $TMP/srv.log"

python3 tests/fault_probe.py --port "$PORT" ${FAULT_CASES:-zombie semantics}
rc=$?

kill -TERM "$SRV_PID" 2>/dev/null
wait "$SRV_PID" 2>/dev/null          # ONLY this pid: never a bare wait
SRV_PID=""
if grep -q "runtime error:\|ERROR: AddressSanitizer" "$TMP/srv.log"; then
    echo "server-faults FAIL: sanitizer finding"; grep "runtime error:\|AddressSanitizer" "$TMP/srv.log" | head; rc=1
fi
[ $rc -eq 0 ] && echo "server-faults OK" || echo "server-faults FAIL"
exit $rc
