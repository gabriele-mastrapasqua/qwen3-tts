#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
O=samples/para_discover; rm -rf $O
M=qwen3-tts-1.7b; SEED=42; SP="${SP:-ryan}"; L="${L:-English}"
dur(){ python3 -c "import wave,sys;w=wave.open(sys.argv[1]);print(f'{w.getnframes()/w.getframerate():.2f}s')" "$1" 2>/dev/null||echo FAIL; }
gen(){ local sub="$1" id="$2" ins="$3" txt="$4"; mkdir -p $O/$sub
  ./qwen_tts -d $M -s $SP -l $L --seed $SEED -T 1.1 -I "$ins" --text "$txt" -o $O/$sub/${id}.wav >/dev/null 2>&1
  echo "  $sub/$id -> $(dur $O/$sub/${id}.wav)"
}
echo "===== PARA DISCOVERY  speaker=$SP lang=$L -> $O ====="

I_COUGH="Cough hard several times, a real chesty cough interrupting the speech."
echo "--- cough ---"
gen cough ahem    "$I_COUGH" "Excuse me, ahem ahem, let me continue now."
gen cough khh     "$I_COUGH" "Excuse me, khh khh, let me continue now."
gen cough coughcough "$I_COUGH" "Excuse me, cough cough, let me continue now."
gen cough zh      "$I_COUGH" "Excuse me, 咳咳, let me continue now."
gen cough ehem    "$I_COUGH" "Excuse me, ehem, let me continue now."

I_SNZ="Sneeze loudly, a sudden uncontrollable sneeze bursting out."
echo "--- sneeze ---"
gen sneeze achoo   "$I_SNZ" "Oh no, achoo, sorry about that."
gen sneeze atchoo  "$I_SNZ" "Oh no, atchoo, sorry about that."
gen sneeze hatchoo "$I_SNZ" "Oh no, hatchoo, sorry about that."
gen sneeze zh      "$I_SNZ" "Oh no, 阿呸, sorry about that."

I_TC="Clear your throat audibly before speaking."
echo "--- throatclear ---"
gen throatclear ahem  "$I_TC" "So, ahem, I would like to say something."
gen throatclear hrm   "$I_TC" "So, hrm hrm, I would like to say something."
gen throatclear ehm   "$I_TC" "So, ehm, I would like to say something."

I_TSK="Click your tongue in disapproval, a sharp tsk sound."
echo "--- tsk ---"
gen tsk tsk   "$I_TSK" "Tsk tsk, you really should not have done that."
gen tsk tch   "$I_TSK" "Tch, you really should not have done that."
gen tsk zh    "$I_TSK" "啲， you really should not have done that."

I_YAWN="Yawn widely and sleepily, a long tired yawn."
echo "--- yawn ---"
gen yawn aah   "$I_YAWN" "I am so tired, aaah, I need to sleep."
gen yawn haaa  "$I_YAWN" "I am so tired, haaaah, I need to sleep."

I_DIS="React with strong disgust, a revolted ugh sound."
echo "--- disgust ---"
gen disgust ugh   "$I_DIS" "Ugh, that is absolutely revolting."
gen disgust eugh  "$I_DIS" "Eugh, that is absolutely revolting."
gen disgust bleh  "$I_DIS" "Bleh, that is absolutely revolting."
echo "DONE -> $O"
