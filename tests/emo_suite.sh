#!/usr/bin/env bash
# ============================================================================================
# emo_suite.sh — ORDERED emotion test suite (per the 2026-06-29 methodology).
#
#   • PER-LANGUAGE subfolders: samples/tests/<DATE>_emo-suite/<lang>/
#   • EMOTION-MATCHED prompts per (language × emotion) — a sentence whose MEANING fits the emotion
#     (Qwen-TTS emotes better on meaningful text than on one neutral carrier reused everywhere).
#   • STEER WEIGHT SWEEP around the sweet spot: w8 / w10 / w12 (w12 wins with the right native speaker — 2026-06-29)
#     PLUS a COMBINE variant (per-language .expr + steer w8 + instruct).
#   • FILENAME encodes voice + mode + steer weight + expr, so you hear the file and KNOW what produced it:
#       <lang>_<voice>_<emo>_steer_w<W>.wav            (pure steer, no expr/instruct)
#       <lang>_<voice>_<emo>_combine_w8_expr.wav       (expr + steer w8 + instruct)
#   • GALATEA clone folder: contrasting emotions across a few languages (COMBINE, the clone cross-lang win).
#
# GOLD voice per language (the recommended one): ryan = IT/EN/PT ; vivian = DE/FR/ES/ZH ; galatea-graft = JA/KO/RU.
# 1.7B CustomVoice, seed 42, ml-range 21-25, ml-decay 0.985 (engine default).
#
# Scope to a subset: LANGS="de fr es" bash tests/emo_suite.sh   (default = all). EMOS="anger sad" to subset emotions.
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
ROOT="${EMO_SUITE_DIR:-samples/tests/2026-06-29_emo-suite}"; mkdir -p "$ROOT"
[ -d "$M" ] || { echo "SKIP: $M not present"; exit 0; }
QL=presets/steer/emotion
WEIGHTS_DEFAULT="8 10 12"   # steer sweep around the w12 sweet spot (2026-06-29 verdict; w12 wins clean)
EMOS="${EMOS:-sad joy anger fear disgust surprise}"
LANGS="${LANGS:-it en de fr es zh ja ko ru pt}"

tok(){ case "$1" in anger)echo ang;; *)echo "$1";; esac; }

# English instruct per emotion (for COMBINE; pure-STEER uses none).
declare -A INS=(
 [sad]="Speak in a sad, sorrowful, gloomy and downcast tone, voice low and heavy, on the verge of tears."
 [joy]="Speak with bright, radiant joy, light and warm, smiling through every word."
 [anger]="Speak in a furious, seething, enraged tone, voice sharp and hard, barely holding back the rage."
 [fear]="Speak in a frightened, trembling, anxious tone, voice shaky and breathless with dread."
 [disgust]="Speak with deep disgust and revulsion, lip-curling contempt, as if something repels you."
 [surprise]="Speak with sudden astonishment and surprise, gasping and caught off guard.")

# language config: tag -> "Language|voiceflags|voicelabel|exprfile"
declare -A LCFG=(
 [it]="Italian|-s ryan|ryan|presets/expr/italian_csp_topk6.expr"
 [en]="English|-s ryan|ryan|presets/expr/italian_csp_topk6.expr"
 [de]="German|-s vivian|vivian|presets/expr/german_csp_k6.expr"
 [fr]="French|-s vivian|vivian|presets/expr/french_csp_k6.expr"
 [es]="Spanish|-s vivian|vivian|presets/expr/italian_csp_topk6.expr"
 [zh]="Chinese|-s vivian|vivian|presets/expr/italian_csp_topk6.expr"
 [ja]="Japanese|--load-voice voices/galatea_graft.qvoice --icl-only|galatea|presets/expr/italian_csp_topk6.expr"
 [ko]="Korean|--load-voice voices/galatea_graft.qvoice --icl-only|galatea|presets/expr/italian_csp_topk6.expr"
 [ru]="Russian|--load-voice voices/galatea_graft.qvoice --icl-only|galatea|presets/expr/italian_csp_topk6.expr"
 [pt]="Portuguese|-s ryan|ryan|presets/expr/italian_csp_topk6.expr")

# EMOTION-MATCHED prompts: "tag_emo" -> sentence whose meaning fits the emotion.
declare -A TX=(
 [it_sad]="Ho perso tutto quello che avevo, e adesso non so più cosa fare."
 [it_joy]="Non ci posso credere, è la notizia più bella della mia vita!"
 [it_anger]="Come ti permetti di parlarmi così? Questo non lo accetto!"
 [it_fear]="C'è qualcuno in casa, ho sentito dei passi... ho davvero paura."
 [it_disgust]="Ma che roba è questa? Fa davvero schifo, non riesco neanche a guardarla."
 [it_surprise]="Cosa?! Non me lo aspettavo per niente, è incredibile!"
 [en_sad]="I've lost everything I had, and now I don't know what to do."
 [en_joy]="I can't believe it, this is the best news of my whole life!"
 [en_anger]="How dare you talk to me like that? I will not accept this!"
 [en_fear]="There's someone in the house, I heard footsteps... I'm really scared."
 [en_disgust]="What is this stuff? It's absolutely revolting, I can't even look at it."
 [en_surprise]="What?! I did not see that coming at all, this is unbelievable!"
 [de_sad]="Ich habe alles verloren, was ich hatte, und jetzt weiß ich nicht mehr weiter."
 [de_joy]="Ich kann es nicht glauben, das ist die schönste Nachricht meines Lebens!"
 [de_anger]="Wie kannst du es wagen, so mit mir zu reden? Das akzeptiere ich nicht!"
 [de_fear]="Da ist jemand im Haus, ich habe Schritte gehört... ich habe Angst."
 [de_disgust]="Was ist das denn? Das ist wirklich widerlich, ich kann gar nicht hinsehen."
 [de_surprise]="Was?! Damit habe ich überhaupt nicht gerechnet, das ist unglaublich!"
 [fr_sad]="J'ai tout perdu, et maintenant je ne sais plus quoi faire."
 [fr_joy]="Je n'arrive pas à y croire, c'est la plus belle nouvelle de ma vie!"
 [fr_anger]="Comment oses-tu me parler comme ça? Je n'accepte pas ça!"
 [fr_fear]="Il y a quelqu'un dans la maison, j'ai entendu des pas... j'ai peur."
 [fr_disgust]="Mais c'est quoi ça? C'est vraiment dégoûtant, je ne peux pas regarder."
 [fr_surprise]="Quoi?! Je ne m'y attendais pas du tout, c'est incroyable!"
 [es_sad]="Lo he perdido todo, y ahora no sé qué hacer."
 [es_joy]="No me lo puedo creer, ¡es la mejor noticia de mi vida!"
 [es_anger]="¿Cómo te atreves a hablarme así? ¡Esto no lo acepto!"
 [es_fear]="Hay alguien en la casa, he oído pasos... tengo miedo."
 [es_disgust]="¿Pero qué es esto? Es asqueroso, ni siquiera puedo mirarlo."
 [es_surprise]="¿Qué?! No me lo esperaba para nada, ¡es increíble!"
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
 [ko_surprise]="뭐?! 전혀 예상 못 했어, 이건 정말 믿을 수 없어!"
 [ru_sad]="Я потерял всё, что у меня было, и теперь не знаю, что мне делать."
 [ru_joy]="Не могу поверить, это лучшая новость в моей жизни, я так счастлив!"
 [ru_anger]="Как ты смеешь так со мной разговаривать? Это неприемлемо!"
 [ru_fear]="В доме кто-то есть, я слышал шаги... мне очень страшно."
 [ru_disgust]="Что это такое? Это отвратительно, я даже смотреть не могу."
 [ru_surprise]="Что?! Я совсем этого не ожидал, это невероятно!"
 [pt_sad]="Perdi tudo o que tinha, e agora não sei o que fazer."
 [pt_joy]="Não posso acreditar, é a melhor notícia da minha vida!"
 [pt_anger]="Como te atreves a falar comigo assim? Não aceito isto!"
 [pt_fear]="Está alguém em casa, ouvi passos... tenho medo."
 [pt_disgust]="Mas o que é isto? É nojento, nem consigo olhar."
 [pt_surprise]="O quê?! Não estava nada à espera, é inacreditável!")

run(){ local out="$1"; shift; if "$@" -o "$out" >"${out%.wav}.log" 2>&1; then echo "  ok $(basename "$out")"; else echo "  FAIL $(basename "$out")"; tail -2 "${out%.wav}.log"; fi; }

for tag in $LANGS; do
  cfg="${LCFG[$tag]:-}"; [ -z "$cfg" ] && { echo "skip unknown lang $tag"; continue; }
  IFS='|' read -r lang vf vlabel expr <<< "$cfg"
  d="$ROOT/$tag"; mkdir -p "$d"
  echo "==== $lang ($vlabel) -> $d ===="
  for e in $EMOS; do
    txt="${TX[${tag}_${e}]:-}"; [ -z "$txt" ] && continue
    ql="$QL/ryan_$(tok $e).qlsteer"
    # steer sweep (pure steer, emotion-matched text, no expr/instruct)
    for w in $WEIGHTS_DEFAULT; do
      run "$d/${tag}_${vlabel}_${e}_steer_w${w}.wav" \
        $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --ml-steer "$ql" --ml-weight $w --ml-range 21-25 --text "$txt"
    done
    # combine (per-language expr @1.2 + steer w8 + instruct)
    run "$d/${tag}_${vlabel}_${e}_combine_w8_expr.wav" \
      $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --expr "$expr" --expr-weight 1.2 \
      --ml-steer "$ql" --ml-weight 8 --ml-range 21-25 --instruct "${INS[$e]}" --text "$txt"
  done
done

# ---- REFERENCE CLONES (CC0/PD, downloadable via download_voices.sh) — the clone recipe = COMBINE everywhere.
#      Reproducible for anyone who clones the repo (galatea IT, quijote ES, ohenry EN, hugo FR). The "one easy
#      way" for clones: IT expr @1.0 (renders) + ryan_<emo> steer w8 + English instruct. ----
gd="$ROOT/reference_clones"; mkdir -p "$gd"
echo "==== reference clones (CC0 25MB grafts) — COMBINE in each native language -> $gd ===="
IT_EXPR=presets/expr/italian_csp_topk6.expr
# voice-file | native-tag | Language
for rv in "galatea_graft|it|Italian" "quijote_graft|es|Spanish" "ohenry_graft|en|English" "hugo_graft|fr|French"; do
  IFS='|' read -r vfile tg lng <<< "$rv"
  if [ ! -f "voices/$vfile.qvoice" ]; then echo "  SKIP $vfile (run: bash download_voices.sh)"; continue; fi
  vn="${vfile%_graft}"; GAL="--load-voice voices/$vfile.qvoice --icl-only"
  for e in anger sad joy; do
    txt="${TX[${tg}_${e}]}"; ql="$QL/ryan_$(tok $e).qlsteer"
    run "$gd/${vn}_${tg}_${e}_combine_w8.wav" \
      $BIN -d $M $GAL -l "$lng" -T 1.1 --seed $SEED --expr "$IT_EXPR" --expr-weight 1.0 \
      --ml-steer "$ql" --ml-weight 8 --ml-range 21-25 --instruct "${INS[$e]}" --text "$txt"
  done
done
# galatea cross-language bonus (the §8.6 win: an IT clone emoting in far languages)
if [ -f voices/galatea_graft.qvoice ]; then
  GAL="--load-voice voices/galatea_graft.qvoice --icl-only"
  for pair in "zh|Chinese|sad" "ru|Russian|anger" "ja|Japanese|joy"; do
    IFS='|' read -r tg lng e <<< "$pair"
    txt="${TX[${tg}_${e}]}"; ql="$QL/ryan_$(tok $e).qlsteer"
    run "$gd/galatea_${tg}_${e}_combine_w8.wav" \
      $BIN -d $M $GAL -l "$lng" -T 1.1 --seed $SEED --expr "$IT_EXPR" --expr-weight 1.0 \
      --ml-steer "$ql" --ml-weight 8 --ml-range 21-25 --instruct "${INS[$e]}" --text "$txt"
  done
fi

echo ""; echo "Done -> $ROOT  ($(find "$ROOT" -name '*.wav' | wc -l | tr -d ' ') clips, per-language subfolders)"
echo "Listen one language: for f in $ROOT/de/*.wav; do echo \$f; afplay \$f; done"
