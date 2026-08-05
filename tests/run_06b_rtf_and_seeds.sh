#!/bin/bash
# E6 batteria 2 — (A) RTF sub-1.0 con lo stack COMPLETO  (B) sweep dei seed para NATIVO sul 0.6B
#
# (B) esiste perché le costanti di para_pick (onomatopea × seed × temperatura) sono tarate sul
# 1.7B — vedi memoria feedback_06b_constants_not_transferable. Sul piccolo un tag che esce male
# NON è un KO finché non hai fatto lo sweep. Qui l'onomatopea la passiamo LETTERALE nel testo e
# variamo --seed, così misuriamo il tag senza le costanti del grosso di mezzo.
#
# ⚠️ Lanciare a MACCHINA SCARICA: l'RTF è la metrica, non il wall-clock.
set -u
cd "$(dirname "$0")/.."
D=samples/tests/2026-08-05_06b_rtf_seeds
M=samples/tests/2026-08-05_06b_full_matrix
mkdir -p "$D/rtf" "$D/seeds"
SMALL=qwen3-tts-0.6b
SEED=42
EVAL="Non è possibile che succeda sempre la stessa cosa, ogni singola volta."
VOICE="$M/voices/galatea_anger.bin"          # la voce emotiva prodotta dalla batteria 1
GRAFT="$M/voices/galatea_anger_graft.qvoice"

say() { echo; echo "=============== $* ==============="; }

# ── A. RTF: lo stack completo scende sotto 1.0? ────────────────────────────────────
say "A. RTF con voce EMOTIVA (4KB) — ladder di quantizzazione, -j4"
for q in "" "--int8" "--quant-mixed" "--int4"; do
  lbl=$(echo "${q:-bf16}" | tr -d '-')
  ./qwen_tts -d $SMALL --load-voice "$VOICE" --xvector-only $q \
    -l Italian --seed $SEED --text "$EVAL" -o "$D/rtf/emo_${lbl}.wav" 2>&1 \
    | grep -E "Audio:" | sed "s/^/  emo ${lbl}: /"
done

say "B. RTF con voce emotiva GRAFT (16.8MB)"
for q in "" "--int8"; do
  lbl=$(echo "${q:-bf16}" | tr -d '-')
  ./qwen_tts -d $SMALL --load-voice "$GRAFT" --icl-only $q \
    -l Italian --seed $SEED --text "$EVAL" -o "$D/rtf/graft_${lbl}.wav" 2>&1 \
    | grep -E "Audio:" | sed "s/^/  graft ${lbl}: /"
done

say "C. RTF con TUTTO ATTIVO: voce emotiva + tag para"
for q in "" "--int8"; do
  lbl=$(echo "${q:-bf16}" | tr -d '-')
  ./qwen_tts -d $SMALL --load-voice "$VOICE" --xvector-only $q \
    -l Italian --seed $SEED --text "[sigh] Non è possibile che succeda sempre la stessa cosa." \
    -o "$D/rtf/all_${lbl}.wav" 2>&1 | grep -E "Audio:" | sed "s/^/  emo+para ${lbl}: /"
done

say "D. riferimento: 0.6B nudo (nessuna voce, nessun tag)"
for q in "" "--int8"; do
  lbl=$(echo "${q:-bf16}" | tr -d '-')
  ./qwen_tts -d $SMALL -s ryan -l Italian --seed $SEED $q --text "$EVAL" \
    -o "$D/rtf/bare_${lbl}.wav" 2>&1 | grep -E "Audio:" | sed "s/^/  nudo ${lbl}: /"
done

# ── B. sweep seed/onomatopea per la para, NATIVO sul piccolo ───────────────────────
# --no-compose passa il testo letterale: niente para_pick, niente costanti del 1.7B.
say "E. sweep para NATIVO 0.6B — onomatopea letterale x seed (voce clonata neutra)"
NEU="$M/voices/galatea_neutral.bin"
for onom_lbl in "sigh:唉" "laugh:哈哈哈" "yawn:哈啊" "wow:哇"; do
  tag="${onom_lbl%%:*}"; onom="${onom_lbl##*:}"
  for s in 7 42 2024 123; do
    ./qwen_tts -d $SMALL --load-voice "$NEU" --xvector-only --no-compose \
      -l Italian --seed $s -T 1.1 --text "${onom}, Non è possibile che succeda sempre la stessa cosa." \
      -o "$D/seeds/${tag}_s${s}.wav" 2>&1 | grep -E "Audio:" | sed "s/^/  ${tag} seed=${s}: /"
  done
done

say "DONE — rtf: $(ls "$D/rtf" | wc -l | tr -d ' ') file · seeds: $(ls "$D/seeds" | wc -l | tr -d ' ') file"
