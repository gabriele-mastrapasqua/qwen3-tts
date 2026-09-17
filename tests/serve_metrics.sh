#!/usr/bin/env bash
# serve_metrics.sh — the metrics endpoint publishes what the process knows, and nothing else.
#
# What this guards, in order of how expensive the mistake would be:
#   1. the flag is refused outside server mode, and refused on the service port;
#   2. a plain single server publishes IDENTITY ONLY — it maintains no request counters, and
#      a page of zeros that look like measurements is worse than an absent series;
#   3. a batched server publishes counters, and they MOVE with real traffic;
#   4. without --metrics-port nothing listens at all;
#   4b. a client polling too fast is refused with 429 and the refusals are counted on the page,
#      and --metrics-max-rate 0 turns that off;
#   5. on Linux, the prefork PARENT publishes one series set per worker, the sets are
#      distinct, and every counter advances under real concurrent traffic.
#
# Block 5 is skipped off Linux, where prefork is compiled out — so on macOS this script
# proves everything except the path that matters most in production. Run it on the box.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-qwen3-tts-0.6b}"
PORT="${PORT:-8944}"
MPORT="${MPORT:-9944}"
BIN=./qwen_tts
FAIL=0

say()  { printf '  %-58s %s\n' "$1" "$2"; }
ok()   { say "$1" "ok"; }
bad()  { say "$1" "FAIL — $2"; FAIL=1; }

cleanup() { pkill -9 -f "qwen_tts.*--serve" >/dev/null 2>&1; }
trap cleanup EXIT
cleanup; sleep 1

start_server() {   # $1 = extra flags; waits for /v1/health
    $BIN -d "$MODEL" --serve "$PORT" $1 >/tmp/serve_metrics.log 2>&1 &
    disown 2>/dev/null   # the cleanup pkill is expected; do not report it as a crash
    for _ in $(seq 1 90); do
        timeout 2 curl -s "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}
# One retry: a single-sample assertion against a network endpoint is a flaky assertion
# regardless of cause, and this one caught a real RST race exactly once in three hundred.
scrape() {
    local body
    body=$(timeout 3 curl -s "http://127.0.0.1:$MPORT/metrics" 2>/dev/null)
    [ -n "$body" ] || body=$(timeout 3 curl -s "http://127.0.0.1:$MPORT/metrics" 2>/dev/null)
    printf '%s' "$body"
}

echo "=== metrics endpoint ==="

# 1. refused where it would publish nothing
$BIN -d "$MODEL" --metrics-port "$MPORT" --text "x" >/dev/null 2>&1
[ $? -ne 0 ] && ok "--metrics-port without --serve is refused" \
             || bad "--metrics-port without --serve is refused" "it was accepted"

$BIN -d "$MODEL" --serve "$PORT" --metrics-port "$PORT" >/dev/null 2>&1
[ $? -ne 0 ] && ok "--metrics-port on the service port is refused" \
             || bad "--metrics-port on the service port is refused" "it was accepted"

# 2. plain single server: identity only, never zero-valued counters
if start_server "--metrics-port $MPORT"; then
    page=$(scrape)
    case "$page" in
        *qwen_tts_build_info*) ok "plain server publishes build identity" ;;
        *) bad "plain server publishes build identity" "no build_info in the page" ;;
    esac
    case "$page" in
        *qwen_tts_worker_completed_total*)
            bad "plain server omits request counters" "it published counters it does not maintain" ;;
        *) ok "plain server omits request counters" ;;
    esac
else
    bad "plain server starts" "no /v1/health within 90s"
fi
cleanup; sleep 1

# 3. batched server: counters exist and move with traffic
if start_server "--batch-size 4 --metrics-port $MPORT"; then
    before=$(scrape | awk -F' ' '/^qwen_tts_worker_completed_total/ {print $2}')
    timeout 120 curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/tts" \
        -H 'Content-Type: application/json' \
        -d '{"text":"metrics regression probe","speaker":"ryan","seed":42,"temperature":0}'
    after=$(scrape | awk -F' ' '/^qwen_tts_worker_completed_total/ {print $2}')
    if [ -n "${before:-}" ] && [ -n "${after:-}" ]; then
        ok "batched server publishes request counters"
        [ "$after" -gt "$before" ] \
            && ok "completed_total advances with traffic ($before -> $after)" \
            || bad "completed_total advances with traffic" "stuck at $before"
    else
        bad "batched server publishes request counters" "completed_total absent"
    fi
    scrape | grep -q '^qwen_tts_worker_slots{worker="0"} 4$' \
        && ok "worker_slots reports the configured batch size" \
        || bad "worker_slots reports the configured batch size" "not 4"
else
    bad "batched server starts" "no /v1/health within 90s"
fi
cleanup; sleep 1

# 4. off by default
if start_server "--batch-size 4"; then
    [ -z "$(scrape)" ] && ok "nothing listens without --metrics-port" \
                       || bad "nothing listens without --metrics-port" "the port answered"
else
    bad "default server starts" "no /v1/health within 90s"
fi

# 4b. rate limit: spam is refused, cheaply, and says so
cleanup; sleep 1
status() { timeout 3 curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$MPORT/metrics" 2>/dev/null; }
burst()  { local n=$1 code hits=0; for _ in $(seq 1 "$n"); do code=$(status);            [ "$code" = "429" ] && hits=$((hits+1)); done; echo "$hits"; }

if start_server "--batch-size 4 --metrics-port $MPORT --metrics-max-rate 2"; then
    hits=$(burst 30)
    [ "$hits" -gt 0 ] && ok "fast polling is refused with 429 ($hits/30)" \
                      || bad "fast polling is refused with 429" "none refused in 30"
    # the counter is on the page, so a served scrape can report how many were turned away
    sleep 2
    n=$(scrape | awk -F' ' '/^qwen_tts_metrics_throttled_total/ {print $2}')
    { [ -n "${n:-}" ] && [ "$n" -gt 0 ]; } \
        && ok "refusals are counted on the page ($n)" \
        || bad "refusals are counted on the page" "throttled_total is '${n:-absent}'"
    # and a scrape that IS served is still a complete page
    sleep 2
    lines=$(scrape | grep -c '^qwen_tts_')
    [ "${lines:-0}" -ge 8 ] && ok "a served scrape is still a complete page ($lines series)" \
                            || bad "a served scrape is still a complete page" "only ${lines:-0} series"
else
    bad "rate-limited server starts" "no /v1/health within 90s"
fi
cleanup; sleep 1

if start_server "--batch-size 4 --metrics-port $MPORT --metrics-max-rate 0"; then
    hits=$(burst 30)
    [ "$hits" -eq 0 ] && ok "--metrics-max-rate 0 disables the limit" \
                      || bad "--metrics-max-rate 0 disables the limit" "$hits/30 still refused"
else
    bad "unlimited server starts" "no /v1/health within 90s"
fi

# 5. prefork parent: one series set per worker, all of them moving (Linux only)
if [ "$(uname -s)" = "Linux" ]; then
    cleanup; sleep 1
    W="${W:-2}"
    if start_server "--prefork $W --batch-size 2 --metrics-port $MPORT"; then
        page=$(scrape)
        seen=$(printf '%s\n' "$page" | grep -c '^qwen_tts_worker_up{worker=')
        [ "$seen" -eq "$W" ] && ok "prefork publishes one up-series per worker ($W)" \
                             || bad "prefork publishes one up-series per worker ($W)" "saw $seen"
        printf '%s\n' "$page" | grep -q '^qwen_tts_rejected_total{reason="all_workers_full"}' \
            && ok "prefork publishes the parent's reject reasons" \
            || bad "prefork publishes the parent's reject reasons" "absent"

        # enough concurrent work that the parent must use every worker
        for _ in $(seq 1 $((W * 3))); do
            timeout 180 curl -s -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/tts" \
                -H 'Content-Type: application/json' \
                -d '{"text":"prefork metrics probe under concurrent load","speaker":"ryan","seed":42,"temperature":0}' &
        done
        wait

        page=$(scrape)
        moved=0; total=0
        for w in $(seq 0 $((W - 1))); do
            d=$(printf '%s\n' "$page" | awk -F' ' -v w="$w" \
                '$1 == "qwen_tts_worker_dispatched_total{worker=\"" w "\"}" {print $2}')
            [ -n "${d:-}" ] || d=0
            total=$((total + d))
            [ "$d" -gt 0 ] && moved=$((moved + 1))
        done
        [ "$moved" -eq "$W" ] && ok "every worker dispatched something ($moved/$W)" \
                              || bad "every worker dispatched something" "only $moved/$W moved"
        [ "$total" -ge $((W * 3)) ] && ok "dispatched_total sums to the offered load ($total)" \
                                    || bad "dispatched_total sums to the offered load" \
                                           "sum $total < $((W * 3)) sent"

        # a counter that goes backwards is the one bug a scraper cannot survive
        a=$(scrape | awk -F' ' '/^qwen_tts_worker_completed_total/ {s+=$2} END {print s+0}')
        b=$(scrape | awk -F' ' '/^qwen_tts_worker_completed_total/ {s+=$2} END {print s+0}')
        [ "$b" -ge "$a" ] && ok "counters are monotonic across scrapes ($a -> $b)" \
                          || bad "counters are monotonic across scrapes" "$a -> $b went backwards"
    else
        bad "prefork server starts" "no /v1/health within 90s"
    fi
else
    say "prefork parent (Linux only)" "skipped on $(uname -s)"
fi

echo
[ "$FAIL" -eq 0 ] && { echo "PASS: metrics endpoint"; exit 0; }
echo "FAIL: metrics endpoint"; exit 1
