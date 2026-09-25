#!/bin/sh
# Provoked failures against a live batched server, and the invariants each must keep:
# model work stops for a client that is gone, the request ends exactly once in the
# session books, a neighbour's audio does not change, RSS does not grow with the aborts,
# and a mixed workload moves the outcome counters by exactly that workload.
#
# The assertions live in tests/fault_probe.py, next to the bytes that provoke them;
# this script only starts the servers, runs the probe and reports.
#
# Usage: tests/test_server_faults.sh [model_dir] [port]
#   FAULT_CASES="zombie semantics service books"   which probe cases (default: all)
#   FAULT_KEEP=1                                     keep the server logs
#   QWEN_CANCEL_ON_DISCONNECT, and any other QWEN_* in the environment, reach the servers.
# Exit: 0 ok, 1 fail, 77 skip (model or binary missing).
MODEL_DIR="${1:-qwen3-tts-0.6b}"
PORT="${2:-8871}"
BIN="${BIN:-./qwen_tts}"
CASES="${FAULT_CASES:-zombie semantics service books}"
[ -f "$MODEL_DIR/config.json" ] || { echo "server-faults SKIP: no model at $MODEL_DIR"; exit 77; }
[ -x "$BIN" ] || { echo "server-faults SKIP: no binary $BIN"; exit 77; }

TMP=$(mktemp -d "${TMPDIR:-/tmp}/qwen_tts_faults.XXXXXX") || exit 1
SRV_PID=""
cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    [ "${FAULT_KEEP:-0}" = 1 ] && echo "server logs kept in $TMP" || rm -rf "$TMP"
}
trap cleanup EXIT

# start_server <log> <port> "<VAR=value ...>" <args...>: sets SRV_PID, 1 if never ready
start_server() {
    log=$1; port=$2; envs=$3; shift 3
    echo "server-faults: $envs $BIN -d $MODEL_DIR --serve $port $*" \
         "(QWEN_CANCEL_ON_DISCONNECT=${QWEN_CANCEL_ON_DISCONNECT-unset})"
    env $envs QWEN_LIFE_TRACE=1 QWEN_REQ_TRACE=1 "$BIN" -d "$MODEL_DIR" --serve "$port" "$@" \
        > "$log" 2>&1 &
    SRV_PID=$!
    for i in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:$port/v1/health" 2>/dev/null | grep -q '"ok"'; then
            echo "server-faults: pid $SRV_PID ready"; return 0
        fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 0.5
    done
    echo "server-faults FAIL: server never became ready"; tail -30 "$log"; return 1
}

# stop_server <log>: SIGTERM, wait for THIS pid only, scan the log
stop_server() {
    kill -TERM "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null          # never a bare wait
    SRV_PID=""
    if grep -q "runtime error:\|ERROR: AddressSanitizer" "$1"; then
        echo "server-faults FAIL: sanitizer finding in $1"
        grep "runtime error:\|AddressSanitizer" "$1" | head; return 1
    fi
    return 0
}

rc=0
MAIN_CASES=$(echo "$CASES" | tr ' ' '\n' | grep -v '^books$' | tr '\n' ' ')
if [ -n "$(echo $MAIN_CASES)" ]; then
    # Batching is pinned bit-exact (the test-batch-invariance pins) so the neighbour's
    # audio can only change through the request lifecycle, never through the kernels a
    # different batch composition picks.
    if start_server "$TMP/srv.log" "$PORT" "QWEN_BATCH_NO_SOLO=1 QWEN_BATCH_NOMATMUL=1" \
                    --batch-size 4 --max-queue 1; then
        python3 tests/fault_probe.py --port "$PORT" --server-pid "$SRV_PID" $MAIN_CASES || rc=1
        stop_server "$TMP/srv.log" || rc=1
    else
        rc=1
    fi
fi

if echo "$CASES" | grep -qw books; then
    BPORT=$((PORT + 1)); MPORT=$((PORT + 2))
    if start_server "$TMP/books.log" "$BPORT" "QWEN_MAX_REQUEST_S=3 QWEN_BATCH_MAX_FRAMES=600" \
                    --batch-size 2 --max-queue 1 --metrics-port "$MPORT"; then
        python3 tests/fault_probe.py --port "$BPORT" --metrics-port "$MPORT" books || rc=1
        stop_server "$TMP/books.log" || rc=1
    else
        rc=1
    fi
fi

[ $rc -eq 0 ] && echo "server-faults OK" || echo "server-faults FAIL"
exit $rc
