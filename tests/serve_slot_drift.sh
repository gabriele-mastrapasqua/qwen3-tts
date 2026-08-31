#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1

if [ -z "${QWEN_BENCH_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
    export QWEN_BENCH_CAFFEINATED=1
    exec caffeinate -i -s "$0" "$@"
fi

MODEL="${MODEL:-qwen3-tts-0.6b}"
SPK="${SPK:-ryan}"
LANG_="${LANG_:-english}"
QUANT="${QUANT:---quant-mixed-int6=q4n14}"
BANK="${BANK:-tests/load_texts_en.txt}"
CLASSES="${CLASSES:-short,medium}"
BATCH="${BATCH:-4}"
REQS="${REQS:-4}"
J="${J:-4}"
PORT="${PORT:-8917}"
SEQ="${SEQ:-1 2 4 1}"
ARMS="${ARMS:-base}"
REQ_TIMEOUT="${REQ_TIMEOUT:-180}"
TEMP="${TEMP:-0}"          # 0 = greedy (il discriminante). 0.9 = come il bench di ieri
OUT="${OUT:-/tmp/tts/slot_drift}"
PY="${PY:-python3}"
BIN=./qwen_tts

[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
[ -d "$MODEL" ] || { echo "modello assente: $MODEL  (./download_model.sh --model small)"; exit 1; }
if pgrep -f "qwen_tts .*--serve" >/dev/null 2>&1; then
    echo "a qwen_tts --serve is already running: stop it, or the measurement is void."; exit 1
fi

rm -rf "$OUT"; mkdir -p "$OUT"
kill_server() { pkill -f "qwen_tts .*--serve $PORT" >/dev/null 2>&1 || true; sleep 1; }
trap kill_server EXIT

ncpu=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "?")
echo "modello   $MODEL ($QUANT) · voce $SPK · $LANG_"
echo "server    --batch-size $BATCH -j $J   ($ncpu core logici)"
echo "sequence  c = $SEQ   ($REQS requests per phase, temperature=$TEMP, fixed seed)"
echo "bracci    $ARMS"

rss_of() {  # RSS in MB del server, portabile
    local pid; pid=$(pgrep -f "qwen_tts .*--serve $PORT" | head -1)
    [ -n "$pid" ] && ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.2f", $1/1024}' || echo "?"
}

run_arm() {
    local arm="$1"
    local env_desc="" ; local -a envp=()
    case "$arm" in
        base)   env_desc="(tutto acceso)" ;;
        nosolo) envp=(QWEN_BATCH_NO_SOLO=1)      ; env_desc="QWEN_BATCH_NO_SOLO=1" ;;
        nobeff) envp=(QWEN_BATCH_NO_BEFF=1)      ; env_desc="QWEN_BATCH_NO_BEFF=1" ;;
        matvec) envp=(QWEN_BATCH_FORCE_MATVEC=1) ; env_desc="QWEN_BATCH_FORCE_MATVEC=1" ;;
        *) echo "braccio sconosciuto: $arm"; return 1 ;;
    esac
    echo
    echo "######## braccio $arm   $env_desc"
    kill_server
    env "${envp[@]}" $BIN -d "$MODEL" $QUANT --serve $PORT --batch-size "$BATCH" -j "$J" \
        > "$OUT/${arm}_server.log" 2>&1 &
    local up=0
    for _ in $(seq 1 120); do
        curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/v1/health" && { up=1; break; }
        sleep 1
    done
    [ "$up" = 1 ] || { echo "  🚨 il server non e' salito — vedi $OUT/${arm}_server.log"; return 1; }
    echo "  su. RSS a modello caricato: $(rss_of) MB"

    local phase=0
    for c in $SEQ; do
        phase=$((phase+1))
        local d="$OUT/${arm}_f${phase}_c${c}"
        $PY tests/load_test.py --url "http://localhost:$PORT" \
            --concurrency "$c" --requests "$REQS" \
            --text-file "$BANK" --classes "$CLASSES" \
            --speaker "$SPK" --language "$LANG_" \
            --temperature "$TEMP" --seed 1000 --timeout "$REQ_TIMEOUT" \
            --output-dir "$d" --csv "$d.csv" >/dev/null 2>&1
        local got; got=$(find "$d" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$got" -lt "$REQS" ]; then
            echo "  🚨 fase $phase (c=$c): solo $got/$REQS clip — timeout o errore"
        else
            echo "  fase $phase (c=$c): ok ($got/$REQS) · RSS $(rss_of) MB"
        fi
    done
    kill_server
}

for a in $ARMS; do run_arm "$a"; done

echo
echo "=============== VERDETTO — fase 1 (c=1) contro l'ultima fase a c=1"
$PY - "$OUT" "$ARMS" "$SEQ" <<'PYEOF'
import sys, os, re, hashlib, wave

out, arms, seq = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
c1 = [i+1 for i, c in enumerate(seq) if c == "1"]
if len(c1) < 2:
    print("sequence has no two c=1 phases: nothing to compare"); sys.exit(0)
first, last = c1[0], c1[-1]

def clips(d):
    """-> {req_id: (audio_s, sha256)}"""
    r = {}
    if not os.path.isdir(d): return r
    for root, _, files in os.walk(d):
        for fn in sorted(files):
            if not fn.endswith(".wav"): continue
            m = re.match(r"req(\d+)_", fn)
            if not m: continue
            p = os.path.join(root, fn)
            with wave.open(p) as w:
                secs = w.getnframes() / float(w.getframerate())
                w.rewind(); pcm = w.readframes(w.getnframes())
            r[int(m.group(1))] = (secs, hashlib.sha256(pcm).hexdigest())
    return r

bad = 0
for arm in arms:
    a = clips(f"{out}/{arm}_f{first}_c1")
    b = clips(f"{out}/{arm}_f{last}_c1")
    ids = sorted(set(a) & set(b))
    print(f"\n{arm}:  phase {first} vs phase {last}   ({len(ids)} comparable requests)")
    if not ids:
        print("  (no clip — the arm failed)"); bad += 1; continue
    print(f"  {'req':>4}{'dur f'+str(first):>14}{'dur f'+str(last):>14}{'delta':>9}{'delta %':>9}  hash")
    worst = 0.0
    for i in ids:
        (sa, ha), (sb, hb) = a[i], b[i]
        d = sb - sa
        pct = (d / sa * 100.0) if sa else float("nan")
        worst = max(worst, abs(pct))
        print(f"  {i:>4}{sa:>14.2f}{sb:>14.2f}{d:>9.2f}{pct:>8.1f}%  {'uguale' if ha==hb else 'DIVERSO'}")
    ident = all(a[i][1] == b[i][1] for i in ids)
    if ident:
        print("  ✅ bit-identical: no state drift on this arm")
    elif worst < 1.0:
        print(f"  🟡 hash diversi ma durate entro {worst:.1f}% — ordine fp, non stato (vedi il caveat B_eff==1)")
    else:
        print(f"  🔴 DRIFT: duration changes by up to {worst:.1f}% after the c=2/c=4 phases"); bad += 1

print("\nRIPRODOTTA" if bad else "\nNON riprodotta su questi bracci")
PYEOF
echo
echo "log dei server: $OUT/*_server.log"
