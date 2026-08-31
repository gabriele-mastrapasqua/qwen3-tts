#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

MODEL=${MODEL:-qwen3-tts-1.7b}
TAG=${TAG:-sigh}
ONOMS=${ONOMS:-"唉,ahh"}
SEEDS=${SEEDS:-"7,42,2024"}
VOICES=${VOICES:-"ryan"}
LANGS=${LANGS:-"English,Italian"}
EMO=${EMO:-}                 # empty -> default per tag below
TEMP=${TEMP:-1.1}
NAME=${NAME:-explore}
JUDGE=${JUDGE:-0}
BIN=${BIN:-./qwen_tts}
DATE=$(date +%F)
OUTDIR=${OUTDIR:-samples/tests/${DATE}_${TAG}_${NAME}}

if [ -z "$EMO" ]; then case "$TAG" in
  sigh) EMO=sad;; laugh) EMO=joy;; cry) EMO=sad;; gasp) EMO=surprise;;
  groan) EMO=disgust;; yawn) EMO=none;; moan) EMO=joy;; throat) EMO=none;; *) EMO=none;; esac; fi

carrier() { # $1=tag $2=lang
  case "$1:$2" in
    sigh:Italian)  echo "Ho perso tutto quello che avevo, %s e adesso non so più cosa fare.";;
    sigh:*)        echo "I lost everything I had, %s and now I don't know what to do.";;
    laugh:Italian) echo "Non ci posso credere, %s, è la notizia più bella della mia vita!";;
    laugh:*)       echo "I can't believe it, %s, this is the best news of my whole life!";;
    cry:Italian)   echo "Mi manca da morire, %s non riesco proprio a smettere.";;
    cry:*)         echo "I miss her so much, %s I just can't stop.";;
    gasp:Italian)  echo "Aspetta, %s ma sei davvero tu?!";;
    gasp:*)        echo "Wait, %s is that really you?!";;
    yawn:Italian)  echo "È talmente tardi, %s non riesco a tenere gli occhi aperti.";;
    yawn:*)        echo "It's so late, %s I can barely keep my eyes open.";;
    groan:Italian) echo "Oh no, %s di nuovo no, che disastro.";;
    groan:*)       echo "Oh no, %s not again, this is awful.";;
    moan:Italian)  echo "Mmm, %s è davvero delizioso, che bontà.";;
    moan:*)        echo "Mmm, %s this tastes absolutely delicious.";;
    throat:Italian) echo "Dunque, %s posso avere la vostra attenzione, prego.";;
    throat:*)      echo "Ahem, %s may I have your attention, please.";;
    *:Italian)     echo "E poi, %s, la storia è andata proprio così.";;
    *)             echo "And then, %s, the story went exactly like that.";;
  esac
}

[ -x "$BIN" ] || { echo "no qwen_tts binary at $BIN (run: make blas)"; exit 1; }
[ -d "$MODEL" ] || { echo "no model dir $MODEL"; exit 1; }
mkdir -p "$OUTDIR"
ROWS="$OUTDIR/.rows.jsonl"; : > "$ROWS"

IFS=',' read -r -a ONOM_A <<< "$ONOMS"
IFS=',' read -r -a SEED_A <<< "$SEEDS"
IFS=',' read -r -a VOICE_A <<< "$VOICES"
IFS=',' read -r -a LANG_A <<< "$LANGS"

n=0
for voice in "${VOICE_A[@]}"; do
  if [[ "$voice" == clone:* ]]; then
    vpath="${voice#clone:}"; vflag=(--load-voice "$vpath" --icl-only); vlabel="clone_$(basename "$vpath" .qvoice)"
  else
    vflag=(-s "$voice"); vlabel="$voice"
  fi
  for lang in "${LANG_A[@]}"; do
    for oi in "${!ONOM_A[@]}"; do
      onom="${ONOM_A[$oi]}"
      for seed in "${SEED_A[@]}"; do
        txt=$(printf "$(carrier "$TAG" "$lang")" "$onom")
        wav="$OUTDIR/${TAG}_${vlabel}_${lang}_o${oi}_s${seed}.wav"
        emoflag=(); [ -n "$EMO" ] && [ "$EMO" != none ] && emoflag=(--emotion "$EMO")
        echo ">> [$((++n))] $vlabel/$lang onom='$onom' seed=$seed emo=${EMO:-none}"
        "$BIN" -d "$MODEL" "${vflag[@]}" -l "$lang" -T "$TEMP" --seed "$seed" \
               "${emoflag[@]}" --text "$txt" -o "$wav" --silent 2>/dev/null || { echo "  gen FAILED"; continue; }
        printf '{"file": "%s", "tag": "%s", "onom": "%s", "seed": %s, "voice": "%s", "lang": "%s", "emo": "%s"}\n' \
               "$(cd "$(dirname "$wav")" && pwd)/$(basename "$wav")" "$TAG" "$onom" "$seed" "$vlabel" "$lang" "$EMO" >> "$ROWS"
      done
    done
  done
done

MANIFEST="$OUTDIR/manifest.json"
python3 - "$ROWS" "$MANIFEST" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
json.dump(rows, open(sys.argv[2], "w"), ensure_ascii=False, indent=2)
print(f"[manifest] {len(rows)} clips -> {sys.argv[2]}")
PY
rm -f "$ROWS"
echo "[done] $n gens in $OUTDIR"

if [ "$JUDGE" = "1" ]; then
  JUDGE_TAGGER=${JUDGE_TAGGER:-cnn14}   # sigh=cnn14 (trusted); laugh -> pass JUDGE_TAGGER=clap or both
  echo "[judge] tagger=$JUDGE_TAGGER"
  tools/para/.venv/bin/python tools/para/para_judge.py \
    --manifest "$MANIFEST" --tagger "$JUDGE_TAGGER" --device cpu \
    --out "$OUTDIR/judge_report.md"
  echo "[judge] -> $OUTDIR/judge_report.md"
fi
