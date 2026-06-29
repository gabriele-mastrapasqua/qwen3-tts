#!/usr/bin/env bash
# ============================================================================================
# perlang_emo_compare.sh — RE-LISTEN set for the per-language emotion recipes extracted from the
# validated notes/scripts (2026-06-29). DIRECT commands (NOT the --emotion router, which is still
# Italian-only) so we hear the REAL per-language GOLD before encoding it. seed 42, 1.7B CustomVoice.
#
# NOTE: the per-language *_r32/_r64 packs (DE/FR/ES) were deleted in a prior asset-hygiene and are
# unrecoverable without retraining → DE/FR use the shipped native *_csp_k6 only (no r32 to compare).
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
OUT=samples/tests/2026-06-29_perlang-emo; mkdir -p "$OUT"
IT=presets/expr/italian_csp_topk6.expr
DE=presets/expr/german_csp_k6.expr
FR=presets/expr/french_csp_k6.expr
QL=presets/steer/emotion
GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
INST_anger="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
INST_sad="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
INST_joy="Speak with bright, radiant joy, light and warm, smiling through every word."
tok(){ case "$1" in anger)echo ang;; *)echo "$1";; esac; }
say(){ printf '  %-28s\n' "$1"; }

# EXPR-only:  out  voiceflags  lang  expr  exprw  emo  "text"
EXPR(){ local o="$1" vf="$2" lang="$3" ex="$4" ew="$5" e="$6" t="$7"; local I="INST_$e"
  say "$o"; $BIN -d $M $vf -l "$lang" -T "${8:-1.1}" --seed $SEED --expr "$ex" --expr-weight "$ew" \
    --instruct "${!I}" --text "$t" -o "$OUT/$o" >"$OUT/${o%.wav}.log" 2>&1 || echo "    FAIL $o"; }
# STEER-only: out voiceflags lang emo "text"  (no instruct)
STEER(){ local o="$1" vf="$2" lang="$3" e="$4" t="$5"
  say "$o"; $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --ml-steer "$QL/ryan_$(tok $e).qlsteer" \
    --ml-weight 8 --ml-range 21-25 --text "$t" -o "$OUT/$o" >"$OUT/${o%.wav}.log" 2>&1 || echo "    FAIL $o"; }
# COMBINE: out voiceflags lang exprw emo "text"  (IT expr + ryan steer + instruct)
COMB(){ local o="$1" vf="$2" lang="$3" ew="$4" e="$5" t="$6"; local I="INST_$e"
  say "$o"; $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --expr "$IT" --expr-weight "$ew" \
    --ml-steer "$QL/ryan_$(tok $e).qlsteer" --ml-weight 8 --ml-range 21-25 \
    --instruct "${!I}" --text "$t" -o "$OUT/$o" >"$OUT/${o%.wav}.log" 2>&1 || echo "    FAIL $o"; }

echo "== DE — vivian + german_csp_k6, EXPR (r32 GONE, k6 only) =="
EXPR de_viv_anger_k6.wav "-s vivian" German "$DE" 1.2 anger "Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."
EXPR de_viv_sad_k6.wav   "-s vivian" German "$DE" 1.2 sad   "Also, lass mich dir in Ruhe erklaeren, wie die Dinge wirklich stehen."

echo "== FR — vivian + french_csp_k6, EXPR =="
EXPR fr_viv_sad_k6.wav   "-s vivian" French "$FR" 1.2 sad   "Bon, laisse-moi t'expliquer calmement comment les choses se passent vraiment."
EXPR fr_viv_anger_k6.wav "-s vivian" French "$FR" 1.2 anger "Bon, laisse-moi t'expliquer calmement comment les choses se passent vraiment."

echo "== ES — ryan + italian_csp_topk6 @ w1.6, T1.3 (Romance-transfer recipe) =="
EXPR es_ryan_joy_w16.wav "-s ryan" Spanish "$IT" 1.6 joy "Bueno, dejame explicarte con calma como estan realmente las cosas." 1.3
EXPR es_ryan_sad_w16.wav "-s ryan" Spanish "$IT" 1.6 sad "Bueno, dejame explicarte con calma como estan realmente las cosas." 1.3

echo "== ZH — vivian (Chinese-native), STEER w8 (same-language → max emotion) =="
STEER zh_viv_joy_steer.wav   "-s vivian" Chinese joy   "我简直不敢相信，这是我一生中最好的消息，我太高兴了！"
STEER zh_viv_anger_steer.wav "-s vivian" Chinese anger "你怎么敢这样跟我说话？这我无法接受，太过分了！"

echo "== JA — galatea-graft COMBINE (IT expr@1.0 + ryan steer w8) =="
COMB ja_gal_sad.wav "$GAL" Japanese 1.0 sad "私が持っていたものを全て失って、もうどうすればいいのか分からない。"
COMB ja_gal_joy.wav "$GAL" Japanese 1.0 joy "信じられない、人生で一番いい知らせだ、本当に幸せだ！"

echo "== KO — galatea-graft COMBINE (joy MUST be combine: steer-alone runs away) =="
COMB ko_gal_joy.wav   "$GAL" Korean 1.0 joy   "믿을 수가 없어, 내 인생 최고의 소식이야, 정말 너무 행복해!"
COMB ko_gal_anger.wav "$GAL" Korean 1.0 anger "네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!"

echo "== RU — galatea-graft COMBINE (top cross-lang win) =="
COMB ru_gal_anger.wav "$GAL" Russian 1.0 anger "Как ты смеешь так со мной разговаривать? Это неприемлемо, я этого не потерплю!"
COMB ru_gal_joy.wav   "$GAL" Russian 1.0 joy   "Не могу поверить, это лучшая новость в моей жизни, я так счастлив!"

echo ""; echo "Done -> $OUT"; ls -1 "$OUT"/*.wav | xargs -n1 basename
