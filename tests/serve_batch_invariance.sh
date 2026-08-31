#!/usr/bin/env bash
# serve_batch_invariance.sh — LA STESSA RICHIESTA DEVE DARE LA STESSA COSA,
#                             CHIUNQUE CONDIVIDA IL BATCH.
#
# PERCHE' ESISTE. Il 2026-08-20 il server batchato ha prodotto DUE bug che nessun gate
# prendeva, ed erano lo stesso bug visto da due lati: `bb->act_idx` / `bb->B_eff` — la
# lista di quali slot la proiezione deve calcolare — descrivevano un frame diverso da
# quello in corso, quindi la testa codec non calcolava la colonna dello slot vivo e il
# campionatore leggeva i logit di un'altra richiesta.
#
#   bug 1  scendendo a UN solo slot attivo si prendeva la scorciatoia single-stream, che
#          usciva prima di impacchettare act_idx -> generazione deragliata: rumore, voce e
#          bip mescolati, durata DOPPIA (21.3 s per un testo da 9).
#   bug 2  uno slot ammesso diventa attivo DOPO l'ultimo passo del Talker, quindi al suo
#          primo frame non e' ancora in act_idx -> ~4 s di farfugliato in TESTA, poi la
#          frase giusta (15.4 s per lo stesso testo da 9).
#
# `make test-batch` non li prendeva: confronta il cablaggio batchato contro il
# single-stream a occupazione FISSA. Nessuno esercitava le due TRANSIZIONI:
#
#   A  N -> 1   una richiesta resta sola dopo che le vicine hanno finito
#   B  1 -> N   una richiesta viene ammessa ACCANTO a una gia' in corso
#
# COME E' COSTRUITO — e la prima versione aveva la premessa SBAGLIATA, quindi vale la pena
# scriverla giusta. Avevo assunto che QWEN_BATCH_NOMATMUL=1 fissasse l'aritmetica e che
# quindi ogni differenza residua fosse cablaggio. Falso: NOMATMUL fissa le PROIEZIONI, ma
# la scorciatoia a uno slot chiama un percorso interamente diverso (attenzione
# single-stream invece di quella batchata). Una richiesta che passa parte della vita da
# sola e parte in compagnia cambia LEGITTIMAMENTE ordine di somma a meta' generazione — e'
# scritto nel commento di qwen_batch_talker_step_ragged, ed e' il motivo per cui esiste
# QWEN_BATCH_NO_SOLO. Misurato: sul codice CORRETTO il braccio B esce con la stessa durata
# e byte diversi. Un gate che pretende l'identita' li' dentro fallisce su codice sano.
#
# Quindi DUE PARTI, ognuna con l'asserzione piu' forte che puo' onestamente fare:
#
#   PARTE 1 — cablaggio, identita' BIT A BIT.
#     QWEN_BATCH_NO_SOLO=1 + QWEN_BATCH_NOMATMUL=1 **e l'endpoint OFFLINE /v1/tts**.
#     I primi due fissano Talker e proiezioni; il terzo serve perche' in STREAMING il
#     decoder ConvNet riceve i frame a blocchi la cui dimensione dipende da quando il
#     ciclo passa da quello slot — cioe' dai vicini — e confini di chunk diversi in un
#     decoder convoluzionale danno audio diverso A PARITA' DI LUNGHEZZA (e' la stessa
#     ragione per cui la decodifica amortizzata non e' byte-identica alla one-shot).
#     Con /v1/tts l'utterance si decodifica in un colpo solo: stesso motore di batching,
#     cambia solo COME si restituisce. Cosi' tutte le richieste attraversano
#     ESATTAMENTE gli stessi kernel nello stesso ordine. Li' l'identita' e' dovuta, e
#     qualunque differenza e' contabilita' degli slot. Prende il bug 2 (uno slot ammesso
#     non e' ancora in act_idx quando la testa codec lo legge). NON prende il bug 1, perche'
#     con NO_SOLO la scorciatoia non gira: e' il prezzo del determinismo.
#
#   PARTE 2 — invarianza di DURATA, coi percorsi veri.
#     Nessun pin: la scorciatoia gira, il batching gira, l'ordine di somma puo' divergere.
#     Li' non si puo' pretendere l'identita' dei byte, ma si puo' pretendere che la
#     richiesta produca lo STESSO NUMERO DI FRAME. Entrambi i bug hanno cambiato la durata
#     in modo grossolano (raddoppiata; +4 s di farfugliato in testa; 6.64 -> 5.92 s sul
#     modello OSS), quindi la durata li prende tutti e due e non ha soglie da tarare.
#
# Campionamento greedy e seed fisso in entrambe: nessuna lotteria da mediare.
#
# OPEN WEIGHTS on purpose: the defect this gate looks for is in the server's wiring, not
# in the weights, and a gate that only runs against a checkpoint nobody else has runs
# neither in CI nor on a rented box.
#
# Uso:
#   tests/serve_batch_invariance.sh
#   MODEL=qwen3-tts-1.7b-base tests/serve_batch_invariance.sh
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${MODEL:-qwen3-tts-0.6b}"
SPK="${SPK:-ryan}"
LANG_="${LANG_:-english}"
QUANT="${QUANT:---int8}"
BATCH="${BATCH:-4}"
J="${J:-4}"
PORT="${PORT:-8953}"
# /v1/tts/stream per l'invarianza di durata; /v1/tts (offline, decodifica one-shot) quando
# si pretende l'identita' bit a bit — vedi l'intestazione.
EP="${EP:-/v1/tts/stream}"
OUT="${OUT:-/tmp/tts/batch_invariance}"
BIN=./qwen_tts

# la richiesta sotto esame, e le due vicine che creano le transizioni
# niente apostrofi nei testi: dentro ${VAR:-...} bash tratta la virgoletta singola come
# quoting e lo script non parsa piu'. Una frase riproducibile batte una pittoresca.
R_TEXT="${R_TEXT:-The engineer will reach your side before two in the afternoon, and if he does not show up I can book another one for you.}"
R_SEED="${R_SEED:-4242}"
LONG_TEXT="Please hold the line while I check the account, the payment has not been posted yet but it will show before the evening, and if it still does not appear by Friday I will escalate it for you and call you back."
SHORT_TEXT="One moment please."

[ -x "$BIN" ] || { echo "build first: make blas"; exit 1; }
[ -d "$MODEL" ] || { echo "🚨 modello assente: $MODEL"; exit 1; }
# Interessa un SERVER in conflitto, non una generazione qualunque: il criterio e' la
# porta. (`pgrep -f` qui e' sicuro: la riga di comando di uno script e' `bash <file>`, non
# contiene il pattern — si auto-matcherebbe solo se lo scrivessi in una riga di shell che
# nomina anche il bersaglio.)
if curl -s -m 2 "http://localhost:$PORT/v1/health" 2>/dev/null | grep -q '"ok"'; then
    echo "🚨 qualcosa risponde gia' sulla porta $PORT: fermalo, o il gate non vale."; exit 1
fi
rm -rf "$OUT"; mkdir -p "$OUT"

req() {  # $1 = file di uscita, $2 = testo, $3 = seed
    # ⚠️ il nome del temporaneo si deriva dal FILE DI USCITA, non da $$: in una sotto-shell
    # in background $$ resta il PID del padre, quindi tre req concorrenti scriverebbero
    # tutte sullo stesso corpo JSON e se lo cancellerebbero a vicenda.
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

echo "=== invarianza del batch: la stessa richiesta, vicini diversi ==="
$BIN --caps 2>&1 | grep -E "^  (build|flag attive|  )" | sed "s/^/  /"
echo "  $MODEL $QUANT · $SPK · -j$J --batch-size $BATCH"
echo "  endpoint $EP · pin: ${PIN_ENV:-nessuno (percorsi veri)} · greedy · seed $R_SEED"
echo "  criterio: ${CRITERION:-byte}"
echo

# ⚠️ NON uccidere per NOME del binario. Il Mac di sviluppo ospita piu' sessioni in repo
# diversi, e un `pkill -x qwen_tts` porta via la generazione di qualcun altro (successo il
# 2026-08-20: ammazzato un render di voice-clone di un altro albero). Si uccide SOLO il
# proprio server, per PID. Sul box affittato non cambia niente: e' dedicato.
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
req "$OUT/_warm.pcm" "Warm up." 7      # la prima generazione paga page fault, non e' la macchina

# ── RIFERIMENTO: da sola, nessun vicino ──
req "$OUT/ref.pcm" "$R_TEXT" "$R_SEED"
echo "  riferimento (da sola)              $(dur "$OUT/ref.pcm") s   $(sum_ "$OUT/ref.pcm")"

# ── A · N -> 1: parte insieme a una vicina CORTA, che finisce prima e la lascia sola ──
# ⚠️ MAI `wait` NUDO qui: aspetterebbe TUTTI i job in background della shell, e fra
# quelli c'e' il SERVER avviato con & poco sopra — che non finisce mai. Lo script resterebbe
# fermo per sempre con la CPU a zero (successo il 2026-08-20). Si aspetta sul PID della sola
# vicina che ci interessa.
req "$OUT/_short.pcm" "$SHORT_TEXT" 11 &
F1=$!
sleep 0.2
req "$OUT/a.pcm" "$R_TEXT" "$R_SEED"
wait $F1
echo "  A · resta sola (N->1)              $(dur "$OUT/a.pcm") s   $(sum_ "$OUT/a.pcm")"

# ── B · 1 -> N: una vicina LUNGA e' gia' in corso quando questa viene ammessa ──
req "$OUT/_long.pcm" "$LONG_TEXT" 12 &
F2=$!
sleep 3
req "$OUT/b.pcm" "$R_TEXT" "$R_SEED"
wait $F2
echo "  B · ammessa accanto (1->N)         $(dur "$OUT/b.pcm") s   $(sum_ "$OUT/b.pcm")"

# ── C · entrambe le transizioni nello stesso giro ──
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
# frame esatti, non secondi arrotondati: 24000 campioni al secondo, 2 byte l'uno
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
                echo "  ✅ $arm stessa durata ($fr campioni) e byte identici"
            else
                echo "  ✅ $arm stessa durata ($fr campioni); byte diversi = ordine di somma, atteso qui"
            fi
        else
            echo "  ❌ $arm DURATA DIVERSA: $fr campioni contro $fa ($(dur "$OUT/ref.pcm") s contro $(dur "$OUT/$arm.pcm") s)"
            rc=1
        fi
    fi
done
echo
if [ "$rc" = 0 ]; then
    echo "batch-invariance [${CRITERION:-byte}]: PASS — la composizione del batch non cambia l'uscita."
else
    echo "batch-invariance [${CRITERION:-byte}]: FAIL"
    echo "  Il primo posto dove guardare e' bb->act_idx / bb->B_eff: devono descrivere i slot"
    echo "  di QUESTO frame su ogni via d'uscita (qwen_batch_pack_active), e devono essere"
    echo "  scritti PRIMA della testa codec, non ereditati dal passo del Talker in fondo al ciclo."
    echo "  pcm grezzi in $OUT"
fi
exit $rc
