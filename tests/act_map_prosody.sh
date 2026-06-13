#!/usr/bin/env bash
# Map WHERE general PROSODY (not emotion) lives in the Talker layers, using the existing
# QWEN_ACT_MAP instrument. Two prosodic axes at CONSTANT neutral emotion, per language:
#   - INTONATION: same sentence as statement vs question ("?" only) — no instruct.
#   - PACING:     same sentence + neutral vs slow vs fast instruct (neutral-instruct base
#                 cancels "instruct presence", isolating the pacing/duration shift).
# Then act_map_diff.py gives the per-layer shift profile -> compare to the known emotion
# bands (magnitude mid L06-11, identity late L21-25). If prosody shows up EARLIER, it's a
# partially-distinct band -> the target for a "language-prosody" LoRA vs the emotion one.
#
# Run on M1 (1.7B present). Audio is throwaway; we only want the .qamp fingerprints.
set -uo pipefail
# T0 (greedy, default) = deterministic -> ZERO seed-noise floor, so every contrast shift is
# pure signal (intonation/pacing are prompt-driven, survive greedy). Override with ACT_T=0.9.
D=qwen3-tts-1.7b; S=ryan; SEED=42; T="${ACT_T:-0}"
A="${ACT_DIR:-/tmp/actmap}"; mkdir -p "$A"
JUNK=/tmp/actmap_junk.wav
cap() {  # cap <qamp-name> <lang> <text> [instruct]
  local name="$1" lang="$2" text="$3" instr="${4:-}"
  local args=(-d $D -s $S --seed $SEED -T $T -l "$lang" --text "$text" -o "$JUNK" --silent)
  [ -n "$instr" ] && args+=(--instruct "$instr")
  QWEN_ACT_MAP="$A/$name.qamp" ./qwen_tts "${args[@]}" >/dev/null 2>&1
  echo "  captured $name.qamp"
}

EN="You're taking the early train tomorrow morning"
IT="Prendi il treno presto domani mattina"
I_NEUT="Speak in a plain, neutral tone."
I_SLOW="Speak slowly and deliberately, drawing out every word."
I_FAST="Speak quickly and briskly, rushing through the sentence."

echo "=== capturing EN ==="
cap en_stmt    English "$EN."
cap en_ques    English "$EN?"
cap en_neutI   English "$EN." "$I_NEUT"
cap en_slow    English "$EN." "$I_SLOW"
cap en_fast    English "$EN." "$I_FAST"
echo "=== capturing IT ==="
cap it_stmt    Italian "$IT."
cap it_ques    Italian "$IT?"
cap it_neutI   Italian "$IT." "$I_NEUT"
cap it_slow    Italian "$IT." "$I_SLOW"
cap it_fast    Italian "$IT." "$I_FAST"

echo; echo "########## EN INTONATION (statement -> question) ##########"
python3 tests/act_map_diff.py "$A/en_stmt.qamp" "$A/en_ques.qamp" --labels question --top 8
echo; echo "########## EN PACING (neutral-instruct -> slow, fast) ##########"
python3 tests/act_map_diff.py "$A/en_neutI.qamp" "$A/en_slow.qamp" "$A/en_fast.qamp" --labels slow,fast --top 8
echo; echo "########## IT INTONATION ##########"
python3 tests/act_map_diff.py "$A/it_stmt.qamp" "$A/it_ques.qamp" --labels question --top 8
echo; echo "########## IT PACING ##########"
python3 tests/act_map_diff.py "$A/it_neutI.qamp" "$A/it_slow.qamp" "$A/it_fast.qamp" --labels slow,fast --top 8
echo; echo "########## CROSS-LANGUAGE: is the intonation shift the same EN vs IT? ##########"
python3 tests/act_map_diff.py "$A/en_stmt.qamp" "$A/en_ques.qamp" "$A/it_ques.qamp" --labels en_q,it_q --top 8
echo "(compare the per-layer profiles above to the emotion bands: magnitude mid L06-11, identity late L21-25)"
