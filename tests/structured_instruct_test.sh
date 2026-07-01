#!/usr/bin/env bash
# ============================================================================================
# structured_instruct_test.sh — plan_emo_v3 §8.8: STRUCTURED-TEMPLATE instruct vs free-form.
#
# QUESTION: does a DEFINED-STRUCTURE instruct (slots VoiceStyle/Tone/Pitch/Tempo/Intensity/Expression)
#   (a) emote BETTER / more controllably than our free-form vivid instruct, and
#   (b) unlock controllable PROSODY by prompt (does Tempo:+X% actually speed up? Pitch:higher raise F0?)
#
# ISOLATED A/B: identical COMBINE recipe (k6 expr + ryan_<emo> steer w8 @L21-25 + instruct), same
#   seed/voice/text/expr/steer — the ONLY variable is the instruct TEXT (free-form vs structured).
#   This does NOT touch emo_suite / THE recipe; it writes a NEW gitignored test folder.
#
# 1.7B only (instruct is 1.7B; 0.6B ignores it). Native preset per language; EMOTION-MATCHED sentences.
#
# PART 1 — emotional quality (full suite 4 voices x 6 emo): neutral / free / tmpl per cell.
#          metric = mel-movement vs the SAME sentence rendered flat (lower corr = moved more) + F0/RMS/dur.
# PART 2 — per-slot prosody control (ryan EN, neutral carrier): sweep ONE slot, others neutral.
#          Tempo->dur_s, Pitch->f0_med, Intensity->rms_db, Expression->f0_std. Does the slot DO anything?
#
# Usage:  bash tests/structured_instruct_test.sh           (full)
#         PART=1 bash tests/structured_instruct_test.sh    (only quality A/B)
#         PART=2 bash tests/structured_instruct_test.sh    (only per-slot sweep)
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
ROOT="${STRUCT_DIR:-samples/tests/2026-06-30_structured-instruct}"
P1="$ROOT/part1_quality"; P2="$ROOT/part2_slots"
QL=presets/steer/emotion
EXPR=presets/expr/italian_csp_topk6.expr
PART="${PART:-0}"   # 0 = both
[ -d "$M" ] || { echo "SKIP: $M not present"; exit 0; }
mkdir -p "$P1" "$P2"

tok(){ case "$1" in anger)echo ang;; *)echo "$1";; esac; }
run(){ local out="$1"; shift; if "$@" -o "$out" >"${out%.wav}.log" 2>&1; then echo "  ok $(basename "$out")"; else echo "  FAIL $(basename "$out")"; tail -2 "${out%.wav}.log"; fi; }

# ---- FREE-FORM vivid instruct per emotion (the current baseline, copied from emo_suite) ----
declare -A FREE=(
 [sad]="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
 [joy]="Speak with bright, radiant joy, light and warm, smiling through every word."
 [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
 [fear]="Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread."
 [disgust]="Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you."
 [surprise]="Speak with sudden astonishment and surprise, gasping and caught off guard.")

# ---- STRUCTURED-TEMPLATE instruct per emotion (the thing under test) ----
declare -A TMPL=(
 [sad]="VoiceStyle: deeply_sad. Tone: sorrowful, gloomy, downcast. Pitch: lower. Tempo: -15%. Intensity: low. Expression: heavy, on the verge of tears."
 [joy]="VoiceStyle: happy_excited. Tone: cheerful, radiant, warm. Pitch: higher. Tempo: +15%. Intensity: medium-high. Expression: bright enunciation, smiling voice."
 [anger]="VoiceStyle: furious. Tone: seething, enraged, hard. Pitch: higher. Tempo: +10%. Intensity: high. Expression: sharp, clipped, barely-contained rage."
 [fear]="VoiceStyle: frightened. Tone: anxious, trembling, dread. Pitch: higher. Tempo: +10%. Intensity: medium. Expression: shaky, breathless."
 [disgust]="VoiceStyle: disgusted. Tone: revolted, contemptuous. Pitch: lower. Tempo: -5%. Intensity: medium. Expression: lip-curling, recoiling."
 [surprise]="VoiceStyle: astonished. Tone: startled, caught off guard. Pitch: higher. Tempo: +5%. Intensity: medium-high. Expression: gasping, wide-eyed.")

# ---- EMOTION-MATCHED sentences (native language per voice) ----
declare -A TX=(
 [en_sad]="I've lost everything I had, and now I don't know what to do."
 [en_joy]="I can't believe it, this is the best news of my whole life!"
 [en_anger]="How dare you talk to me like that? I will not accept this!"
 [en_fear]="There's someone in the house, I heard footsteps... I'm really scared."
 [en_disgust]="What is this stuff? It's absolutely revolting, I can't even look at it."
 [en_surprise]="What?! I did not see that coming at all, this is unbelievable!"
 [zh_sad]="我失去了我所拥有的一切，现在我不知道该怎么办。"
 [zh_joy]="我简直不敢相信，这是我一生中最好的消息，我太高兴了！"
 [zh_anger]="你怎么敢这样跟我说话？这我无法接受，太过分了！"
 [zh_fear]="家里有人，我听到了脚步声……我很害怕，不知道该怎么办。"
 [zh_disgust]="这是什么东西？太恶心了，我连看都不想看。"
 [zh_surprise]="什么？！我完全没想到，这太不可思议了！"
 [ja_sad]="私が持っていたものを全て失って、もうどうすればいいのか分からない。"
 [ja_joy]="信じられない、人生で一番いい知らせだ、本当に幸せだ！"
 [ja_anger]="よくも私にそんな口のきき方ができるな！絶対に受け入れられない！"
 [ja_fear]="家に誰かいる、足音が聞こえた……怖くてどうしたらいいか分からない。"
 [ja_disgust]="これは何？本当に気持ち悪い、見ることもできない。"
 [ja_surprise]="えっ？！全く予想していなかった、信じられない！"
 [ko_sad]="내가 가진 모든 걸 잃었어, 이제 어떻게 해야 할지 모르겠어."
 [ko_joy]="믿을 수가 없어, 내 인생 최고의 소식이야, 정말 너무 행복해!"
 [ko_anger]="네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!"
 [ko_fear]="집에 누군가 있어, 발소리를 들었어... 무서워서 어떻게 해야 할지 모르겠어."
 [ko_disgust]="이게 뭐야? 정말 역겨워, 쳐다보기도 싫어."
 [ko_surprise]="뭐?! 전혀 예상 못 했어, 이건 정말 믿을 수 없어!")

# voice config: tag|Language|voiceflag|voicelabel
P1_VOICES=(
 "en|English|-s ryan|ryan"
 "zh|Chinese|-s vivian|vivian"
 "ja|Japanese|-s ono_anna|ono_anna"
 "ko|Korean|-s sohee|sohee")
EMOS="${EMOS:-sad joy anger fear disgust surprise}"

# COMBINE render with a given instruct ("" = none, used for neutral anchor: no expr/steer/instruct)
combine(){ # out lang vf emo instruct
  local out="$1" lang="$2" vf="$3" e="$4" ins="$5" txt="$6"; local ql="$QL/ryan_$(tok "$e").qlsteer"
  if [ -z "$ins" ]; then
    run "$out" $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --text "$txt"
  else
    run "$out" $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --expr "$EXPR" --expr-weight 1.2 \
      --ml-steer "$ql" --ml-weight 8 --ml-range 21-25 --instruct "$ins" --text "$txt"
  fi
}

if [ "$PART" = "0" ] || [ "$PART" = "1" ]; then
  echo "==== PART 1: emotional quality (free-form vs structured), 4 voices x ${EMOS} ===="
  for cfg in "${P1_VOICES[@]}"; do
    IFS='|' read -r tag lang vf vlabel <<< "$cfg"
    for e in $EMOS; do
      txt="${TX[${tag}_${e}]:-}"; [ -z "$txt" ] && continue
      combine "$P1/${tag}_${vlabel}_${e}_neutral.wav" "$lang" "$vf" "$e" "" "$txt"
      combine "$P1/${tag}_${vlabel}_${e}_free.wav"    "$lang" "$vf" "$e" "${FREE[$e]}" "$txt"
      combine "$P1/${tag}_${vlabel}_${e}_tmpl.wav"    "$lang" "$vf" "$e" "${TMPL[$e]}" "$txt"
    done
  done
fi

if [ "$PART" = "0" ] || [ "$PART" = "2" ]; then
  echo "==== PART 2: per-slot prosody control (ryan EN, neutral carrier) ===="
  CARR="The meeting is scheduled for three o'clock in the afternoon on Tuesday."
  VF="-s ryan"; LANG=English
  base(){ echo "VoiceStyle: neutral. Tone: plain and even. Pitch: $1. Tempo: $2. Intensity: $3. Expression: $4."; }
  slot(){ # name instruct
    run "$P2/p2_$1.wav" $BIN -d $M $VF -l "$LANG" -T 1.1 --seed $SEED --expr "$EXPR" --expr-weight 1.2 \
      --ml-steer "$QL/ryan_joy.qlsteer" --ml-weight 8 --ml-range 21-25 --instruct "$2" --text "$CARR"; }
  # anchor: all-neutral template
  slot "anchor"        "$(base neutral +0% medium 'plain enunciation')"
  # Tempo sweep (others neutral) -> expect dur_s to DECREASE as % rises
  slot "tempo_-20"     "$(base neutral -20% medium 'plain enunciation')"
  slot "tempo_+20"     "$(base neutral +20% medium 'plain enunciation')"
  slot "tempo_+40"     "$(base neutral +40% medium 'plain enunciation')"
  # Pitch sweep -> expect f0_med to RISE lower->higher
  slot "pitch_lower"   "$(base lower +0% medium 'plain enunciation')"
  slot "pitch_higher"  "$(base higher +0% medium 'plain enunciation')"
  # Intensity sweep -> expect rms_db to RISE low->high
  slot "intensity_low" "$(base neutral +0% low 'plain enunciation')"
  slot "intensity_high" "$(base neutral +0% high 'plain enunciation')"
  # Expression sweep -> f0_std (variance) + ear
  slot "expr_flat"     "$(base neutral +0% medium 'flat monotone, no inflection')"
  slot "expr_smiling"  "$(base neutral +0% medium 'smiling, warm, lively voice')"
fi

# ---- MEASURE ----
echo "==== measuring ===="
MEAS="$ROOT/metrics.tsv"
python3 tests/measure_prosody.py --header > "$MEAS"
find "$ROOT" -name '*.wav' | sort | while read -r w; do
  python3 tests/measure_prosody.py "$w" >> "$MEAS"
done
echo "  per-file metrics -> $MEAS"

if [ "$PART" = "0" ] || [ "$PART" = "1" ]; then
  MOVE="$ROOT/movement.tsv"
  echo -e "cell\tfree_corr\ttmpl_corr\twinner(lower=moves more)" > "$MOVE"
  for cfg in "${P1_VOICES[@]}"; do
    IFS='|' read -r tag lang vf vlabel <<< "$cfg"
    for e in $EMOS; do
      n="$P1/${tag}_${vlabel}_${e}_neutral.wav"; f="$P1/${tag}_${vlabel}_${e}_free.wav"; t="$P1/${tag}_${vlabel}_${e}_tmpl.wav"
      [ -s "$n" ] && [ -s "$f" ] && [ -s "$t" ] || continue
      fc=$(python3 tests/measure_prosody.py --move "$n" "$f")
      tc=$(python3 tests/measure_prosody.py --move "$n" "$t")
      win=$(python3 -c "print('tmpl' if $tc < $fc else 'free')")
      echo -e "${tag}_${vlabel}_${e}\t${fc}\t${tc}\t${win}" >> "$MOVE"
    done
  done
  echo "  movement (mel-corr vs neutral, lower=moves more) -> $MOVE"
fi

echo ""; echo "DONE -> $ROOT  ($(find "$ROOT" -name '*.wav' | wc -l | tr -d ' ') clips)"
echo "Tables: $ROOT/metrics.tsv  $ROOT/movement.tsv"
