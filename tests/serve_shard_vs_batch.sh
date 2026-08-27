#!/usr/bin/env bash
# serve_shard_vs_batch.sh — UN PROCESSO GROSSO O TANTI PICCOLI?
#
# La domanda che 4 core non permettevano di porre. A parità di CORE TOTALI e di stream
# serviti, due forme di deployment opposte:
#
#   BATCH   1 server -jN con --batch-size S      il batching continuo condivide la LETTURA
#                                                 DEI PESI fra gli slot, ma tutti gli slot
#                                                 avanzano in lockstep: il piu' lento detta
#                                                 il passo a tutti.
#   SHARD   S server -j(N/S), uno per stream      niente lockstep, isolamento vero (un crash
#                                                 non porta via S conversazioni), ma ogni
#                                                 processo rilegge i pesi per conto suo.
#
# PERCHE' NON E' OVVIO. Sulla c3 il batching stravinceva contro il percorso seriale
# (0.6B, c=4: p95 12669 -> 1547 ms), ma quello era un confronto a THREAD TOTALI FISSI su una
# macchina satura, dove i processi separati non avevano core in cui esistere. Con 8 core
# veri ognuno degli S processi ha i suoi, e il lockstep diventa il costo dominante del
# batching invece che il suo prezzo di ammissione.
#
# LA MEMORIA, e c'e' una previsione da falsificare: i pesi grandi sono bf16 MMAPPATI, quindi
# S processi che mappano lo stesso file condividono le stesse pagine fisiche — quella parte
# e' 1x, non Sx. Cio' che si paga S volte e' la copia QUANTIZZATA (int8/int4), che vive
# sull'heap ed e' privata. Atteso: RSS(S) ~ RSS(1) + (S-1) x copia_quant. Se esce Sx RSS(1),
# la previsione e' sbagliata e il loader non condivide come si crede: leggilo, non assumerlo.
#
# COSA SI CONFRONTA, e va tenuto onesto: stesso banco di testi, stessi arrivi (uniform, cosi'
# due corse sono identiche per costruzione e nessun grappolo casuale distingue i bracci),
# stesso numero di richieste TOTALI, stessi core totali. Nel braccio SHARD ogni client parla
# col SUO server (round-robin perfetto = il caso migliore per un bilanciatore); nel braccio
# BATCH gli stessi client parlano tutti con l'unico server.
#
# Uso:
#   tests/serve_shard_vs_batch.sh                          # 0.6B, 3 stream, 8 core
#   MODEL=qwen3-tts-1.7b-base STREAMS=3 CORES=8 tests/serve_shard_vs_batch.sh
#
# Open models only: the guard below refuses a model that lives in a private tree.
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-qwen3-tts-0.6b}"
STREAMS="${STREAMS:-3}"                 # stream concorrenti serviti (= S)
CORES="${CORES:-$(nproc 2>/dev/null || echo 8)}"
QUANT="${QUANT:---int8}"
REQS="${REQS:-8}"                       # richieste PER STREAM
SPK="${SPK:-ryan}"; LANG_="${LANG_:-english}"
BANK="${BANK:-tests/load_texts_en.txt}"
CLASSES="${CLASSES:-short,medium}"
# ⚠️ Il carico offerto deve stare SOTTO la capacita' del braccio piu' lento, altrimenti la
# coda cresce senza limite e il TTFA misura l'attesa in coda invece della latenza: il braccio
# con piu' capacita' vince per costruzione e il confronto non dice niente. Con ~8.5 s di audio
# per richiesta e un braccio batchato che regge Q~1.26, 30 s fra gli arrivi di ogni client
# fanno 0.85 audio-s/s offerti su 3 stream: sotto per entrambi.
INTERVAL="${INTERVAL:-30}"              # s fra gli arrivi di uno stesso client (uniform)
REQ_TIMEOUT="${REQ_TIMEOUT:-180}"
PORT0="${PORT0:-8970}"
OUT="${OUT:-/tmp/tts/shard_vs_batch}"
PY="${PY:-python3}"
BIN=./qwen_tts

# (the model-tree guard that used to live in a wrapper script is inlined below)
case "$MODEL" in *models/*|*outputs/*) echo "refusing a model from a private tree: $MODEL" >&2; exit 1;; esac
[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
mkdir -p "$OUT"

JS=$(( CORES / STREAMS )); [ "$JS" -lt 1 ] && JS=1     # thread per shard
echo "=== un processo grosso o $STREAMS piccoli? ==="
echo "  modello   $MODEL $QUANT · $STREAMS stream · $CORES core totali"
echo "  BATCH     1 server -j$CORES --batch-size $STREAMS"
echo "  SHARD     $STREAMS server -j$JS (uno per stream)"
echo "  carico    $REQS richieste per stream · arrivi uniform ogni ${INTERVAL}s · classi $CLASSES"

kill_all() { pkill -9 -f "qwen_tts.*--serve" 2>/dev/null; sleep 2; }
trap kill_all EXIT

# ⚠️ NON sommare l'RSS dei processi: l'RSS conta le pagine CONDIVISE in OGNI processo che
# le mappa, quindi S processi che condividono gli stessi pesi mmappati sembrano occupare Sx
# la memoria anche quando la fisica occupata e' 1x. E' esattamente l'errore che questa
# misura esiste per evitare (successo il 2026-08-19: 8554 MB "usati" da 3 shard contro 3468
# di un processo solo, cioe' 2.5x, che NON era la memoria reale).
# Due numeri corretti al posto suo:
#   PSS  (Proportional Set Size) divide ogni pagina condivisa fra chi la mappa -> sommabile.
#   USED memoria di sistema realmente occupata, il numero che paga la fattura.
pss_of() {  # somma il PSS di tutti i server, in MB
    local t=0 v
    for pid in $(pgrep -f "qwen_tts.*--serve" 2>/dev/null); do
        v=$(awk '/^Pss:/ {s+=$2} END {print s+0}' "/proc/$pid/smaps_rollup" 2>/dev/null)
        t=$((t + ${v:-0}))
    done
    echo $((t / 1024))
}
used_of() {  # memoria di sistema occupata (MB), esclusa la page cache
    free -m 2>/dev/null | awk '/^Mem:/ {print $3}'
}
# ⏱️ Attesa che il server sia PRONTO, non che il processo esista.
#
# 180 s bastavano per UN server. Non bastano per S server che caricano insieme: sul
# c4a (2026-08-21) il braccio SHARD a S=4 e' morto con "server non partito" mentre i
# quattro stavano ancora caricando — ognuno mmappa 3,8 GB E costruisce la sua copia
# int8 privata, con un quarto dei core a testa. Il log del server lo diceva: rispondeva
# a /v1/health, ma non ancora con "ok". Un banco che dichiara morto cio' che sta
# lavorando produce un buco nei dati che sembra un difetto del prodotto.
#
# Quindi: finestra larga, e soprattutto DIRE cosa si sta aspettando, cosi' un'attesa
# lunga si distingue da un server che non partira' mai.
WAIT_READY_S="${WAIT_READY_S:-600}"
wait_ready() {  # $1 = porta
    local i
    for i in $(seq 1 "$WAIT_READY_S"); do
        curl -s -m 2 "http://localhost:$1/v1/health" 2>/dev/null | grep -q '"ok"' && {
            [ "$i" -gt 30 ] && echo "     porta $1 pronta dopo ${i}s"
            return 0
        }
        case $i in 60|120|240|360|480) echo "     ...porta $1: ancora in caricamento a ${i}s" ;; esac
        sleep 1
    done
    echo "     porta $1: nessun \"ok\" da /v1/health entro ${WAIT_READY_S}s — ultimo stato:"
    curl -s -m 2 "http://localhost:$1/v1/health" 2>/dev/null | head -c 200 | sed 's/^/       /'
    echo
    return 1
}

run_arm() {  # $1 = batch | shard
    local arm="$1" ports="" p n
    kill_all
    if [ "$arm" = batch ]; then
        $BIN -d "$MODEL" $QUANT --serve "$PORT0" --batch-size "$STREAMS" -j "$CORES" \
            > "$OUT/${arm}_srv0.log" 2>&1 &
        wait_ready "$PORT0" || { echo "  🚨 server non partito"; return 1; }
        for n in $(seq 1 "$STREAMS"); do ports="$ports $PORT0"; done
    else
        for n in $(seq 0 $((STREAMS - 1))); do
            p=$((PORT0 + 1 + n))
            # --batch-size anche qui, e NON e' un dettaglio: sotto 2 il server prende il
            # percorso legacy SENZA scheduler continuo (qwen_tts_serve_batched parte da 2),
            # e da quando /v1/health dice la verita' (20/08) quel percorso si dichiara
            # "scheduler: down" -> il banco lo aspettava per 600 s e poi lo dava per morto.
            # Con 2 slot per shard i due bracci girano sullo STESSO codice e l'unica
            # variabile resta la topologia, che e' il punto dell'esperimento.
            $BIN -d "$MODEL" $QUANT --serve "$p" --batch-size 2 -j "$JS" > "$OUT/${arm}_srv$n.log" 2>&1 &
            ports="$ports $p"
            sleep 2   # sfalsati: S processi che aprono lo stesso file insieme si contendono
                      # la page cache e il disco, e il primo carico e' gia' il piu' lento
        done
        for p in $ports; do wait_ready "$p" || { echo "  🚨 server $p non partito"; return 1; }; done
    fi

    # scaldata: la prima generazione paga allocazioni e page fault che non sono la macchina a regime
    for p in $ports; do
        curl -s -m "$REQ_TIMEOUT" -o /dev/null -X POST "http://localhost:$p/v1/tts/stream" \
            -d "{\"text\":\"Warm up the engine before measuring.\",\"speaker\":\"$SPK\",\"language\":\"$LANG_\",\"seed\":7}" || true
    done
    local pss used; pss=$(pss_of); used=$(used_of)

    # un client per stream, tutti insieme; nel braccio shard ognuno sul proprio server
    local n=0 pids=""
    for p in $ports; do
        QWEN_LT_ARRIVAL=uniform QWEN_LT_INTERVAL="$INTERVAL" \
        $PY tests/load_test.py --url "http://localhost:$p" --path /v1/tts/stream \
            --concurrency 1 --requests "$REQS" --text-file "$BANK" --classes "$CLASSES" \
            --speaker "$SPK" --language "$LANG_" --timeout "$REQ_TIMEOUT" --no-save-audio \
            --json "$OUT/${arm}_c$n.json" --csv "$OUT/${arm}_c$n.csv" > /dev/null 2>&1 &
        pids="$pids $!"; n=$((n + 1))
    done
    for pid in $pids; do wait "$pid"; done
    kill_all

    $PY - "$OUT" "$arm" "$STREAMS" "$pss" "$used" <<'PYEOF'
import csv, json, sys, glob, os
out, arm, streams = sys.argv[1], sys.argv[2], int(sys.argv[3])
pss, used = int(sys.argv[4]), int(sys.argv[5])

# I percentili si ricostruiscono dai record PER RICHIESTA (--csv), non dal blocco "spikes"
# del JSON: quello contiene solo le richieste FUORI soglia, quindi darebbe percentili
# sistematicamente pessimisti. E i percentili per-client non si mediano fra loro: si
# rimettono insieme i campioni grezzi e si calcola una volta sola.
ttfa = []
for f in sorted(glob.glob(os.path.join(out, arm + "_c*.csv"))):
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = r.get("ttfa_ms", "")
            if v not in ("", None):
                try:
                    ttfa.append(float(v))
                except ValueError:
                    pass
audio = wall = 0.0
for f in sorted(glob.glob(os.path.join(out, arm + "_c*.json"))):
    d = json.load(open(f))
    for lvl in (d if isinstance(d, list) else [d]):
        if isinstance(lvl, dict):
            audio += lvl.get("audio_s", 0.0)
            wall = max(wall, lvl.get("wall_s", 0.0))
ttfa.sort()

def pct(v, q):
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]

qq = audio / wall if wall else float("nan")
print("  %-6s TTFA p50 %6.0f ms . p95 %6.0f ms . campioni %2d . Q %.2f . PSS %d MB . sistema %d MB"
      % (arm, pct(ttfa, 0.5), pct(ttfa, 0.95), len(ttfa), qq, pss, used))
PYEOF
}

run_arm batch
run_arm shard
echo
echo "  ⚠️  i percentili escono dai record PER RICHIESTA (--csv) di tutti i client messi"
echo "      insieme: il blocco spikes del JSON contiene solo le richieste fuori soglia."
echo "  ⚠️  PSS, non somma di RSS: l'RSS conta le pagine condivise in ogni processo che le"
echo "      mappa. Atteso ~1x i pesi mmappati + Sx la copia quantizzata."
