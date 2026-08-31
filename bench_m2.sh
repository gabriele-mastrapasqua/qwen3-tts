#!/usr/bin/env bash
set -u
BIN=${BIN:-./qwen_tts}
SEED=${SEED:-42}
SPK=${SPK:-ryan}
LNG=${LNG:-Italian}                                   # NB: not $LANG (that's the locale)
TXT=${TXT:-"Ciao, questo e' un test di sintesi vocale per misurare le prestazioni su questa macchina."}
MODELS=${MODELS:-"qwen3-tts-0.6b qwen3-tts-1.7b"}     # only existing dirs are run
PRECS=${PRECS:-"bf16 int8 int4"}                      # bf16=no flag, int8=--int8, int4=--int4
MODES=${MODES:-"single stream"}                       # single=no flag, stream=--stream
BACKENDS=${BACKENDS:-"metal cpu"}                      # cpu = default engine path (no --backend)
OUTDIR=${OUTDIR:-bench_out}
STAMP=$(date +%Y%m%d_%H%M%S 2>/dev/null || echo run)
LOG="$OUTDIR/bench_${STAMP}.log"
SUMMARY="$OUTDIR/summary_${STAMP}.txt"

mkdir -p "$OUTDIR"
: > "$LOG"; : > "$SUMMARY"

say(){ echo "$@" | tee -a "$LOG" >&2; }

say "════════════════════════════════════════════════════════════════════════════"
say " qwen-tts  bench_m2  —  $STAMP"
say "════════════════════════════════════════════════════════════════════════════"
say "binary : $BIN"
say "host   : $(uname -srm 2>/dev/null)"
if command -v sysctl >/dev/null 2>&1; then
  say "chip   : $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo '?')"
  say "cpu    : $(sysctl -n hw.ncpu 2>/dev/null) cores / mem $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0)/1073741824 )) GB"
fi
say ""
say "── --caps (compiled SIMD; note the CPU tier) ──"
"$BIN" --caps 2>&1 | tee -a "$LOG" >&2
CPU_TIER=$("$BIN" --caps 2>&1 | grep -iE "^ *note:" | head -1 | sed -E 's/^ *note: *//')
say ""

say "── GPU self-test (matvec/matmat vs CPU) ──"
"$BIN" --gpu-selftest --backend metal 2>&1 | tee -a "$LOG" | tail -3 >&2 || say "(gpu-selftest not available in this binary)"
say ""

prec_flag(){ case "$1" in bf16) echo "";; int8) echo "--int8";; int4) echo "--int4";; esac; }
mode_flag(){ case "$1" in single) echo "";; stream) echo "--stream";; esac; }

printf "%-14s %-7s %-7s %-7s | %-8s %-8s %-8s\n" model backend prec mode audio_s gen_s RTF | tee -a "$SUMMARY" >&2
printf -- "----------------------------------------------------------------------------\n" | tee -a "$SUMMARY" >&2

for M in $MODELS; do
  [ -d "$M" ] || { say "skip $M (dir missing)"; continue; }
  for BK in $BACKENDS; do
    BKFLAG=""; ENVP=""
    if [ "$BK" = "metal" ]; then BKFLAG="--backend metal"; ENVP="QWEN_METAL_FUSED_TALKER=1"; fi
    for P in $PRECS; do
      for MD in $MODES; do
        RUNLOG="$OUTDIR/${M##*/}_${BK}_${P}_${MD}_${STAMP}.log"
        CMD="$ENVP $BIN -d $M $BKFLAG $(prec_flag "$P") $(mode_flag "$MD") \
             --seed $SEED -s $SPK -l $LNG --text \"$TXT\" -o $OUTDIR/o_${BK}_${P}_${MD}.wav"
        echo ">>> $CMD" >> "$LOG"
        env $ENVP "$BIN" -d "$M" $BKFLAG $(prec_flag "$P") $(mode_flag "$MD") \
            --seed "$SEED" -s "$SPK" -l "$LNG" --text "$TXT" \
            -o "$OUTDIR/o_${BK}_${P}_${MD}.wav" > "$RUNLOG" 2>&1
        RC=$?
        RTFLINE=$(grep -iE "RTF" "$RUNLOG" | tail -1)
        cat "$RUNLOG" >> "$LOG"
        if [ $RC -ne 0 ] || [ -z "$RTFLINE" ]; then
          printf "%-14s %-7s %-7s %-7s | %-8s %-8s %-8s\n" "${M##*/}" "$BK" "$P" "$MD" "-" "-" "ERR(rc=$RC)" | tee -a "$SUMMARY" >&2
        else
          A=$(echo "$RTFLINE" | sed -nE 's/.*Audio: ([0-9.]+)s.*/\1/p')
          G=$(echo "$RTFLINE" | sed -nE 's/.* in ([0-9.]+)s.*/\1/p')
          R=$(echo "$RTFLINE" | sed -nE 's/.*RTF ([0-9.]+).*/\1/p')
          printf "%-14s %-7s %-7s %-7s | %-8s %-8s %-8s\n" "${M##*/}" "$BK" "$P" "$MD" "$A" "$G" "$R" | tee -a "$SUMMARY" >&2
        fi
      done
    done
  done
done

say ""
say "════════════════════════════════════════════════════════════════════════════"
say "CPU tier of this binary: ${CPU_TIER:-unknown}"
say "  → if it says 'M1-class', this is an scp'd M1 binary: the METAL column is a"
say "    true reading of THIS box's GPU, but the CPU column is M1-level. Rebuild"
say "    native (./setup_m2.sh) for the real M2/M4 CPU (i8mm/bf16) numbers."
say "Full log : $LOG"
say "Summary  : $SUMMARY"
say "════════════════════════════════════════════════════════════════════════════"
