#!/usr/bin/env bash
# Emotion seed-finder palette → recommended-seeds doc.
#
# WHAT: for each (language × voice × emotion) cell, render N seeds with --seed-audition N
# --audition-keep (saving EVERY take), capture per-seed glitch+duration + the binary's auto-pick,
# and emit a markdown table of RECOMMENDED seeds (docs/emotion-seeds.md by default). The kept takes
# form a browsable palette; the auto-pick (glitch+duration) is a HEURISTIC — it picks the cleanest/
# most-typical take, NOT necessarily the most expressive one → EAR-VERIFY before promoting a seed to
# the README. (A SER-judge in the loop would rank recognizability; pass JUDGE_MODEL=<dir> if you have
# one — tests/emo_judge.py — else the doc relies on glitch+dur + your ear.)
#
# WHY: a fixed seed can damp emotion (seed42 "easy" vs seed777 "rage", same recipe), and a seed that
# glitches on one (text×voice) is fine on another → there is no global magic seed. This tool surfaces,
# per (lang×voice×emo), the stable+clean candidates so you can curate a shortlist and ship a few as
# README examples that ALSO teach how to drive the emotion FT (.expr) packs.
#
# USAGE: tests/emotion_seed_finder.sh [out_md] [N_seeds]
#   env knobs (all optional):
#     LANGS="Italian Spanish"     which languages
#     VOICES_IT / VOICES_ES       voice list per language (override the defaults)
#     EMOS_IT / EMOS_ES           emotion list per language
#     N=8  BASE_SEED=42           seeds = BASE_SEED .. BASE_SEED+N-1
#     MODEL=qwen3-tts-1.7b  THREADS=4
#     OUTDIR=samples/emotion_seeds   (audio palette; local-only/gitignored, regenerable)
#     JUDGE_MODEL=<ser dir>       optional: add a recognizability column via tests/emo_judge.py
#
# The .expr packs and the 1.7B model are local-only/large; the script SKIPS cleanly if absent.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

OUT_MD="${1:-docs/emotion-seeds.md}"
N="${2:-${N:-5}}"    # 4-5 is enough: a broken seed is obvious in the table, no need for 8
BASE_SEED="${BASE_SEED:-42}"
MODEL="${MODEL:-qwen3-tts-1.7b}"
THREADS="${THREADS:-4}"
OUTDIR="${OUTDIR:-samples/emotion_seeds}"
BIN=./qwen_tts
PACK_WIN="presets/expr/italian_csp.expr"        # 2-block CSP-FT (WIN): voices that already speak the lang
PACK_K4="presets/expr/italian_csp_topk4.expr"   # 4-block CSP-FT (k4): clones / voices that drift

LANGS="${LANGS:-Italian Spanish}"
# Per-language voice + emotion sets (override via env).
VOICES_IT="${VOICES_IT:-ryan vivian galatea}"
VOICES_ES="${VOICES_ES:-ryan vivian}"
EMOS_IT="${EMOS_IT:-neutral anger disgust fear joy sadness surprise}"
EMOS_ES="${EMOS_ES:-anger joy sadness}"     # validated transfer subset

# Neutral, emotion-agnostic carrier sentences (so the INSTRUCT drives the emotion, not the text).
TEXT_IT="${TEXT_IT:-Domani mattina ci vediamo davanti alla stazione.}"
TEXT_ES="${TEXT_ES:-Mañana por la mañana nos vemos frente a la estación.}"

# EMOVO English instructs (model follows instruct best in EN; same strings the FT was steered with).
instruct_for() {
  case "$1" in
    neutral)  echo "" ;;
    joy)      echo "Speak happily, bright and warm, smiling through the words." ;;
    anger)    echo "Speak with hot, furious anger, sharp and forceful." ;;
    sadness)  echo "Speak with a sad, sorrowful, downcast tone, voice low and heavy." ;;
    fear)     echo "Speak with fear, tense and trembling, your voice wary." ;;
    surprise) echo "Speak with surprise, startled and taken aback, held through the whole sentence." ;;
    disgust)  echo "Speak with physical disgust, repulsed and recoiling." ;;
    *)        echo "" ;;
  esac
}

# Voice → qwen args (preset vs clone x-vector .bin).
voice_args() {
  case "$1" in
    ryan)    echo "-s ryan" ;;
    vivian)  echo "-s vivian" ;;
    galatea) echo "--load-voice voices/galatea.bin --xvector-only" ;;
    *)       echo "-s $1" ;;
  esac
}

# Recipe (pack|temp|weight) per (lang, voice, emo) — REVISED from hands-on CLI calibration
# (user 2026-06-21, see [[project_seed_and_onset_emotion]] #4):
#   - k4 (top4k) is the DEFAULT/recommended pack: more robust at high weight than WIN, and vivian
#     reacts with more energy. WIN reachable via env PACK_DEFAULT=$PACK_WIN (cap w≈1.8).
#   - TEMPERATURE is the language-error dial: IT sweet spot T0.5–1.0 (NOT 1.1–1.3). >1.2 eats words,
#     1.8 code-switches to Chinese, 0.1 is deterministic+flat. We use T0.8 for calm emotions,
#     T1.0 for high-arousal (anger/fear/surprise — k4 holds w2.0 @ T1.0 with zero IT errors).
#   - WEIGHT: high-arousal w2.0 (k4 takes it), calm w1.6 (preset) / 1.8 (clone needs a touch more push).
#   - ES (transfer): keep WIN @ T1.0 w1.6 (k4 over-steered ES sad→French in earlier tests; revisit).
PACK_DEFAULT="${PACK_DEFAULT:-$PACK_K4}"
is_high_arousal() { case "$1" in anger|fear|surprise) return 0 ;; *) return 1 ;; esac; }
recipe_for() {
  local lang="$1" voice="$2" emo="$3" pack temp w
  if [ "$lang" = "Spanish" ]; then
    pack="$PACK_WIN"; temp="1.0"; w="1.6"
  else
    pack="$PACK_DEFAULT"
    if is_high_arousal "$emo"; then temp="1.0"; w="2.0";
    else temp="0.8"; [ "$voice" = "galatea" ] && w="1.8" || w="1.6"; fi
  fi
  echo "$pack|$temp|$w"
}

[ -x "$BIN" ] || { echo "Build first: make blas"; exit 1; }
[ -d "$MODEL" ] || { echo "SKIP: model $MODEL not present"; exit 0; }

mkdir -p "$OUTDIR"
TS="$(date '+%Y-%m-%d %H:%M')"
: > "$OUT_MD"

# ---- doc header + how-to-use intro -------------------------------------------------
{
cat <<EOF
# Emotion FT — recommended seeds & usage (palette)

> Generated by \`tests/emotion_seed_finder.sh\` on $TS · model \`$MODEL\` · N=$N seeds (base $BASE_SEED) · auto-pick = glitch+duration heuristic (**ear-verify before trusting**).

## How the emotion fine-tune (\`.expr\`) works

The expressive emotion is a small **characteristic-specific fine-tune** (CSP-FT) shipped as an additive
weight delta you apply on top of any voice at generation time:

- **\`$PACK_K4\`** — 4-block FT ("k4", **recommended default**): more capacity, robust at high weight, reacts with more energy.
- **\`$PACK_WIN\`** — 2-block FT ("WIN"): gentler/lighter; cap the weight around 1.8.

Recipe = **spoken text in the target language + \`-l <Lang>\` + an instruct in ENGLISH + temperature + the \`.expr\` pack at a per-emotion weight**. The instruct is followed best in English even when speaking another language. Spanish reuses the Italian pack (\`--expr … -l Spanish\`) — the FT transfers to close Romance languages.

### The two dials (hands-on calibration)

- **Temperature = the language-cleanliness dial.** Italian sweet spot **T0.5–1.0**. Higher = more emotional variety but more errors: **T1.2 eats words**, **T1.8 leaves Italian / mixes Chinese**. **T≈0.1 is deterministic** (the softmax is near-greedy → the seed has *no* effect, all takes identical) and flat. Use ~**T0.8** for calm emotions, **T1.0** for high-arousal (anger/fear/surprise).
- **Weight = the emotion-strength dial.** Push it up for stronger emotion; **k4 holds w2.0 cleanly at T1.0**, WIN starts breaking past ~1.8. Calm emotions ≈ **w1.6**, high-arousal ≈ **w2.0**.

### Finding a good seed
At temperature the **seed selects a sub-mode** of the emotion; a fixed seed can damp it or, at high weight/temp, *break* it (gibberish, or a second of noise then stop). Use the built-in best-of-N audition — it rejects the broken takes and keeps the cleanest:

\`\`\`bash
$BIN -d $MODEL -s ryan -l Italian -T 1.0 --expr $PACK_K4 --expr-weight 2.0 \\
  --instruct "Speak with hot, furious anger, sharp and forceful." \\
  --seed-audition 5 --audition-keep --text "$TEXT_IT" -o anger.wav
\`\`\`

\`--seed-audition N\` renders N seeds and keeps the cleanest (a take much shorter than the median = truncated, or a metallic runaway tail, is rejected); \`--audition-keep\` also saves **every** take as \`<out>.seed<seed>.wav\` so you can browse and pick by ear. ⚠ in a table flags a take whose duration is out of the plausible band (likely broken). The tables below are that audition across the whole emotion × language × voice matrix.

EOF
} >> "$OUT_MD"

# ---- run the matrix ----------------------------------------------------------------
run_cell() { # $1 lang  $2 voice  $3 emo
  local lang="$1" voice="$2" emo="$3"
  local text varg rec pack temp w instr
  if [ "$lang" = "Spanish" ]; then text="$TEXT_ES"; else text="$TEXT_IT"; fi
  varg="$(voice_args "$voice")"
  rec="$(recipe_for "$lang" "$voice" "$emo")"
  pack="${rec%%|*}"; temp="$(echo "$rec" | cut -d'|' -f2)"; w="${rec##*|}"
  instr="$(instruct_for "$emo")"
  [ -f "$pack" ] || { echo "  SKIP cell $lang/$voice/$emo: pack $pack absent"; return; }

  local cdir="$OUTDIR/${lang,,}/${voice}"; mkdir -p "$cdir"
  local out="$cdir/${emo}.wav"
  local log="$cdir/${emo}.log"

  # Build the command (instruct only when non-empty → neutral anchor has none).
  local -a cmd=("$BIN" -d "$MODEL" -j"$THREADS" -T "$temp" --seed "$BASE_SEED" -l "$lang"
                $varg --expr "$pack" --expr-weight "$w" --seed-audition "$N" --audition-keep
                --text "$text" -o "$out")
  [ -n "$instr" ] && cmd+=(--instruct "$instr")

  echo "  [$lang/$voice/$emo] pack=$(basename "$pack") T$temp w$w ..."
  "${cmd[@]}" >"$log" 2>&1

  # Reconstruct a copy-paste command string for the doc.
  local cmdstr="$BIN -d $MODEL -T $temp --seed $BASE_SEED -l $lang $varg --expr $pack --expr-weight $w"
  [ -n "$instr" ] && cmdstr="$cmdstr --instruct \"$instr\""
  cmdstr="$cmdstr --seed-audition $N --audition-keep --text \"$text\" -o ${emo}.wav"

  local picked; picked="$(grep -E 'picked seed' "$log" | sed -E 's/.*picked seed ([0-9]+).*/\1/' | head -1)"

  # Emit the per-cell markdown.
  {
    echo
    echo "#### ${emo} — \`${voice}\` · $lang"
    if [ -n "$picked" ]; then
      echo "- **Recommended seed: \`$picked\`** *(auto glitch+dur pick — ear-verify)* · pack \`$(basename "$pack")\` · T$temp · weight $w"
    else
      echo "- ⚠️ no take produced (see $log)"
    fi
    [ -n "$instr" ] && echo "- instruct: \"$instr\"" || echo "- instruct: *(none — neutral anchor)*"
    echo '```bash'
    echo "$cmdstr"
    echo '```'
    if grep -qE 'audition seed' "$log"; then
      # Median take-duration for this cell → flag out-of-band takes (⚠ = likely broken).
      local med
      med="$(grep -E 'audition seed' "$log" | sed -E 's/.*: ([0-9.]+)s glitch.*/\1/' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
      echo "| seed | dur (s) | glitch | listen |"
      echo "|---|---|---|---|"
      grep -E 'audition seed' "$log" | while read -r line; do
        local s d g sf
        s="$(echo "$line"  | sed -E 's/.*audition seed ([0-9]+):.*/\1/')"
        d="$(echo "$line"  | sed -E 's/.*: ([0-9.]+)s glitch.*/\1/')"
        g="$(echo "$line"  | sed -E 's/.*glitch=([0-9.]+).*/\1/')"
        sf="$cdir/${emo}.seed${s}.wav"   # relative to repo root → portable in the committed doc
        local mark=""; [ "$s" = "$picked" ] && mark=" ⭐"
        local warn; warn="$(awk -v d="$d" -v m="$med" 'BEGIN{ if(m>0){r=d/m; if(r<0.55||r>2.5) printf "⚠"} }')"
        echo "| $s$mark$warn | $d | $g | \`afplay $sf\` |"
      done
    fi
  } >> "$OUT_MD"
}

echo "=== Emotion seed-finder: model=$MODEL N=$N base=$BASE_SEED → $OUT_MD ==="
for lang in $LANGS; do
  if [ "$lang" = "Spanish" ]; then voices="$VOICES_ES"; emos="$EMOS_ES";
  else voices="$VOICES_IT"; emos="$EMOS_IT"; fi
  echo "## $lang" >> "$OUT_MD"
  for voice in $voices; do
    # x-vector clone needs its .bin; skip the cell cleanly if missing.
    if [ "$voice" = "galatea" ] && [ ! -f voices/galatea.bin ]; then
      echo "  SKIP voice galatea ($lang): voices/galatea.bin absent"; continue; fi
    echo "### Voice: \`$voice\`" >> "$OUT_MD"
    for emo in $emos; do run_cell "$lang" "$voice" "$emo"; done
  done
done

echo "" >> "$OUT_MD"
echo "_Audio palette under \`$OUTDIR/\` (local-only/gitignored — regenerate with \`make emotion-seeds\`)._" >> "$OUT_MD"
echo "=== DONE → $OUT_MD ==="
