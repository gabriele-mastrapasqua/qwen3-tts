#!/usr/bin/env bash
# Focused follow-up (2026-06-29 ear verdicts): STEER wins (w10/w8). Two open items:
#  (1) JA/KO must use the NATIVE PRESET speakers (ono_anna=JA, sohee=KO), NOT the galatea clone.
#  (2) anger/fear: try w12 (does it hold or break?). STEER pure (emotion-matched text, no instruct).
# Out: samples/tests/2026-06-29_native-w12/  (gitignored). 1.7B, seed 42, L21-25.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=./qwen_tts; M=qwen3-tts-1.7b; SEED=42
O=samples/tests/2026-06-29_native-w12; mkdir -p "$O"
QL=presets/steer/emotion
tok(){ case "$1" in anger)echo ang;; *)echo "$1";; esac; }
S(){ local out="$1" vf="$2" lang="$3" e="$4" w="$5" txt="$6"
  if $BIN -d $M $vf -l "$lang" -T 1.1 --seed $SEED --ml-steer "$QL/ryan_$(tok $e).qlsteer" \
       --ml-weight $w --ml-range 21-25 --text "$txt" -o "$O/$out" >"$O/${out%.wav}.log" 2>&1; then
    echo "  ok $out"; else echo "  FAIL $out"; tail -2 "$O/${out%.wav}.log"; fi; }

# emotion-matched prompts
JA_anger="よくも私にそんな口のきき方ができるな！絶対に受け入れられない！"
JA_sad="私が持っていたものを全て失って、もうどうすればいいのか分からない。"
JA_joy="信じられない、人生で一番いい知らせだ、本当に幸せだ！"
JA_fear="家に誰かいる、足音が聞こえた……怖くてどうしたらいいか分からない。"
KO_anger="네가 어떻게 나한테 그렇게 말할 수 있어? 이건 절대 받아들일 수 없어!"
KO_sad="내가 가진 모든 걸 잃었어, 이제 어떻게 해야 할지 모르겠어."
KO_joy="믿을 수가 없어, 내 인생 최고의 소식이야, 정말 너무 행복해!"
KO_fear="집에 누군가 있어, 발소리를 들었어... 무서워서 어떻게 해야 할지 모르겠어."
DE_anger="Wie kannst du es wagen, so mit mir zu reden? Das akzeptiere ich nicht!"
DE_fear="Da ist jemand im Haus, ich habe Schritte gehört... ich habe Angst."
EN_anger="How dare you talk to me like that? I will not accept this!"
EN_fear="There's someone in the house, I heard footsteps... I'm really scared."
ES_anger="¿Cómo te atreves a hablarme así? ¡Esto no lo acepto!"
ES_fear="Hay alguien en la casa, he oído pasos... tengo miedo."

echo "== (1) JA on ono_anna (native preset), STEER w10 + anger/fear w12 =="
S ja_ono_anna_anger_steer_w10.wav "-s ono_anna" Japanese anger 10 "$JA_anger"
S ja_ono_anna_anger_steer_w12.wav "-s ono_anna" Japanese anger 12 "$JA_anger"
S ja_ono_anna_fear_steer_w10.wav  "-s ono_anna" Japanese fear  10 "$JA_fear"
S ja_ono_anna_fear_steer_w12.wav  "-s ono_anna" Japanese fear  12 "$JA_fear"
S ja_ono_anna_sad_steer_w10.wav   "-s ono_anna" Japanese sad   10 "$JA_sad"
S ja_ono_anna_joy_steer_w10.wav   "-s ono_anna" Japanese joy   10 "$JA_joy"

echo "== (1) KO on sohee (native preset), STEER w10 + anger/fear w12 =="
S ko_sohee_anger_steer_w10.wav "-s sohee" Korean anger 10 "$KO_anger"
S ko_sohee_anger_steer_w12.wav "-s sohee" Korean anger 12 "$KO_anger"
S ko_sohee_fear_steer_w10.wav  "-s sohee" Korean fear  10 "$KO_fear"
S ko_sohee_fear_steer_w12.wav  "-s sohee" Korean fear  12 "$KO_fear"
S ko_sohee_sad_steer_w10.wav   "-s sohee" Korean sad   10 "$KO_sad"
S ko_sohee_joy_steer_w10.wav   "-s sohee" Korean joy   10 "$KO_joy"

echo "== (2) anger/fear w12 on DE/EN/ES (does w12 hold or break?) =="
S de_vivian_anger_steer_w12.wav "-s vivian" German  anger 12 "$DE_anger"
S de_vivian_fear_steer_w12.wav  "-s vivian" German  fear  12 "$DE_fear"
S en_ryan_anger_steer_w12.wav   "-s ryan"   English anger 12 "$EN_anger"
S en_ryan_fear_steer_w12.wav    "-s ryan"   English fear  12 "$EN_fear"
S es_vivian_anger_steer_w12.wav "-s vivian" Spanish anger 12 "$ES_anger"
S es_vivian_fear_steer_w12.wav  "-s vivian" Spanish fear  12 "$ES_fear"

echo ""; echo "Done -> $O ($(ls "$O"/*.wav 2>/dev/null | wc -l | tr -d ' ') clips)"
