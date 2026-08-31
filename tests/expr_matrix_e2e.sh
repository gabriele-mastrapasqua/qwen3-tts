#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-/tmp/e2e}"; mkdir -p "$OUT"
M=qwen3-tts-1.7b
EXPR=presets/expr/italian_bb027_ep5_r32.expr
W=0.6; T=1.1; SEED=42; LANG=Italian; ICL=voices/galatea_icl.qvoice
HAP="Speak with warm, bright happiness, smiling through the words."
SHORT="Che bella notizia, sono davvero felicissimo oggi!"
LONG="Oggi è una giornata splendida. Voglio raccontarti una storia che mi è rimasta nel cuore. Era un pomeriggio d'autunno e tutto sembrava sospeso nel tempo."
TBL="$OUT/listen.txt"; : > "$TBL"
note(){ printf "%-30s %s\n" "$1" "$2" | tee -a "$TBL"; }

gen(){ # $1 label  $2 voice-args  $3 text  $4..=extra
  local label="$1" varg="$2" txt="$3"; shift 3
  ./qwen_tts -d $M --text "$txt" --seed $SEED -l $LANG -T $T --expr $EXPR --expr-weight $W -I "$HAP" $varg "$@" -o "$OUT/$label.wav" --silent 2>/dev/null
  note "$OUT/$label.wav" "$label"
}

echo "### SINGLE — preset ryan + clone galatea_icl, short + long"
gen ryan_short  "-s ryan"           "$SHORT"
gen ryan_long   "-s ryan"           "$LONG"
gen clone_short "--load-voice $ICL"  "$SHORT"
gen clone_long  "--load-voice $ICL"  "$LONG"

echo "### STREAM (long)"
gen ryan_long_stream  "-s ryan"          "$LONG" --stream
gen clone_long_stream "--load-voice $ICL" "$LONG" --stream

echo "### AUDIOBOOK --batch (long multi-sentence -> chunked)"
gen ryan_long_batch  "-s ryan"          "$LONG" --batch --batch-words 10
gen clone_long_batch "--load-voice $ICL" "$LONG" --batch --batch-words 10

echo "### SERVER request-batching (2 concurrent, ryan, LoRA baked)"
PORT=8124
./qwen_tts -d $M --serve $PORT --expr $EXPR --expr-weight $W -s ryan -l $LANG -T $T --batch-size 2 > "$OUT/server.log" 2>&1 &
SRV=$!
until curl -s http://localhost:$PORT/v1/health >/dev/null 2>&1; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; break; }; done
curl -s -X POST http://localhost:$PORT/v1/tts -d "{\"text\":\"$SHORT\",\"seed\":42,\"speaker\":\"ryan\",\"language\":\"Italian\",\"instruct\":\"$HAP\"}" -o "$OUT/server_req1.wav" & P1=$!
curl -s -X POST http://localhost:$PORT/v1/tts -d "{\"text\":\"$LONG\",\"seed\":7,\"speaker\":\"ryan\",\"language\":\"Italian\",\"instruct\":\"$HAP\"}" -o "$OUT/server_req2.wav" & P2=$!
wait $P1 $P2            # wait ONLY on the curls, NOT the server (which never exits)
kill $SRV 2>/dev/null
note "$OUT/server_req1.wav" "server_req1 (short)"
note "$OUT/server_req2.wav" "server_req2 (long)"

echo "=== DONE. durations ==="
python3 -c "import soundfile as sf,glob,os
for w in sorted(glob.glob('$OUT/*.wav')):
    try: i=sf.info(w); print(f'{os.path.basename(w):24s} {round(i.frames/i.samplerate,2)}s')
    except Exception as e: print(os.path.basename(w),'ERR',e)"