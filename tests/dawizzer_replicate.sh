#!/usr/bin/env bash
cd "$(dirname "$0")/.."
OUT=samples/dawizzer
mkdir -p "$OUT"
M=qwen3-tts-1.7b-base
REF=samples/voice_clone_refs/it_galatea_fasol.wav
RT="Notizie mie, eccole. Sono venuto qua, come sai, per dar pace a questi poveri nervi, e ci lavoro a lacrimente chiudendomi nell'inerzia più fitta."
TEXT="Non riesco proprio a crederci. Come hai potuto farmi questo?"
S=42

EMOS=(
  "neutral 1.0 0.8 1.05"
  "angry   1.3 0.85 1.15"
  "sad     0.8 0.7 1.0"
  "excited 1.4 0.9 1.20"
)

run() { # mode name T p r  extra...
  local mode=$1 name=$2 T=$3 p=$4 r=$5; shift 5
  echo ">> $mode/$name  T=$T p=$p r=$r"
  timeout 220 ./qwen_tts -d "$M" --ref-audio "$REF" --seed "$S" --max-duration 25 \
    -l Italian -T "$T" -p "$p" -r "$r" -k 50 --text "$TEXT" \
    -o "$OUT/${mode}_${name}.wav" --silent "$@" 2>/dev/null
  [ $? -eq 124 ] && echo "   !! TIMEOUT"
  local sz=$(stat -f%z "$OUT/${mode}_${name}.wav" 2>/dev/null || echo 0)
  local dur=$(python3 -c "print(f'{$sz/2/24000:.1f}')" 2>/dev/null)
  echo "   ${mode}_${name}.wav  ${dur}s ${sz}B"
}

echo "=== FAST mode (x_vector_only) — his fast path ==="
for e in "${EMOS[@]}"; do set -- $e; run fast "$1" "$2" "$3" "$4" --xvector-only; done

echo "=== ACCURATE mode (ICL, ref_text) — his accurate path (= our ICL; the hang suspect) ==="
for e in "${EMOS[@]}"; do set -- $e; run icl "$1" "$2" "$3" "$4" --ref-text "$RT"; done
echo "ALL DONE -> $OUT  (dur ~25s = EOS hang)"
