#!/usr/bin/env bash
# ============================================================================
# x86_bench.sh — clean check + RTF/TTFA/batching A/B on an x86 box.
# "git pull && bash tests/x86_bench.sh <model>" — no fragile copy-paste pipes.
#
# Sections:
#   [0] CORRECTNESS gate — --caps (does VNNI fire?) + --self-test (kernels correct
#       on THIS ISA, incl. the batched int8/q4 VNNI matmat twins). MUST pass first.
#   [A] Same-core (-j1): does the kernel work help? scalar-bf16 vs VNNI-int8.
#   [B] Full single-stream RTF matrix (-j4): scalar/avx2/VNNI × bf16/int8/int4/quant-mixed,
#       incl. VNNI-on/off A/B and the int4 SDOT-q4 (C7) on/off A/B.
#   [C] Batched matmat (--matmat-bench): the server-throughput kernel win (weight read
#       ONCE for B streams). int8/q4 now use VNNI (the batched-no-VNNI bug is fixed).
#   [D] TTFA (streaming): ms to first audio chunk, per precision.
#   [E] (optional) e2e server request-batching — set BATCH=1 to run it.
#
# Usage:  bash tests/x86_bench.sh                 # uses qwen3-tts-0.6b
#         bash tests/x86_bench.sh qwen3-tts-1.7b  # 1.7B (int4/quant-mixed are 1.7B)
#         BATCH=1 bash tests/x86_bench.sh qwen3-tts-1.7b   # + [E] server batching
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MODEL="${1:-qwen3-tts-0.6b}"
TXT="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."

if [ ! -d "$MODEL" ]; then
    echo "Model dir '$MODEL' not found. Pass one: bash tests/x86_bench.sh <model-dir>"
    exit 1
fi

# Build a SIMD level into a named binary if not already present.
build() { # $1=SIMD  $2=outname
    if [ -x "$2" ]; then echo ">> reuse $2"; return 0; fi
    echo ">> building $2 (SIMD=$1) ..."
    make clean >/dev/null 2>&1
    if make blas SIMD="$1" >/tmp/build_$1.log 2>&1; then
        cp -f qwen_tts "$2"; echo "   ok ($(du -h "$2" | cut -f1))"
    else
        echo "   BUILD FAILED (SIMD=$1) — tail:"; tail -8 /tmp/build_$1.log; return 1
    fi
}
build scalar     qwen_tts_scalar
build avx2       qwen_tts_avx2
build avx512vnni qwen_tts_avx512vnni

run() { # $1=label  then the command + its flags
    local label="$1"; shift
    local out rtf cp
    out=$("$@" -d "$MODEL" --text "$TXT" --seed 42 -s ryan -l English -o /tmp/bench.wav 2>&1)
    rtf=$(printf '%s\n' "$out" | grep -oE 'RTF [0-9.]+' | head -1 | awk '{print $2}')
    cp=$(printf '%s\n'  "$out" | grep -oE '[0-9.]+ ms/f' | tail -1)
    printf "  %-34s RTF %-7s CP %s\n" "$label" "${rtf:-ERR}" "${cp:-?}"
}
ttfa() { # $1=label then command+flags
    local label="$1"; shift
    local t
    t=$("$@" -d "$MODEL" --text "$TXT" --seed 42 -s ryan -l English --stream -o /tmp/bench.wav 2>&1 | grep -oE 'TTFA: [0-9]+ ms')
    printf "  %-34s %s\n" "$label" "${t:-ERR}"
}

echo "================================================================"
echo " x86 bench — $MODEL"
echo " CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
echo " vCPUs: $(nproc) | L3-share(cpu0): $(cat /sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list 2>/dev/null)"
echo "================================================================"
echo "[0] CORRECTNESS — does VNNI fire + are the kernels correct?"
./qwen_tts_avx512vnni --caps 2>&1 | grep -iE 'int8 dot|runtime cpu|lever'
echo -n "    self-test: "; ./qwen_tts_avx512vnni --self-test 2>&1 | grep -oE 'SELF-TEST (PASSED|FAILED).*'
echo
echo "[A] Same core count (-j1): does the kernel work help?"
run "scalar bf16 -j1 (~original)" ./qwen_tts_scalar               -j1
run "VNNI   int8 -j1 (ours)"      ./qwen_tts_avx512vnni  --int8    -j1
run "VNNI   int4 -j1 (C7)"        ./qwen_tts_avx512vnni  --int4    -j1
echo
echo "[B] Single-stream RTF matrix (-j4):"
run "scalar bf16 -j4"             ./qwen_tts_scalar                -j4
run "avx2   int8 -j4"             ./qwen_tts_avx2        --int8    -j4
run "VNNI   int8 -j4"             ./qwen_tts_avx512vnni  --int8    -j4
run "VNNI   int8 -j4 (vnni OFF)"  env QWEN_NO_VNNI=1 ./qwen_tts_avx512vnni --int8 -j4
run "VNNI   int4 -j4 (C7 SDOT-q4)" ./qwen_tts_avx512vnni --int4    -j4
run "VNNI   int4 -j4 (SDOT OFF)"  env QWEN_NO_SDOT=1 ./qwen_tts_avx512vnni --int4 -j4
run "VNNI   quant-mixed -j4"      ./qwen_tts_avx512vnni  --quant-mixed -j4
run "VNNI   bf16 -j4"             ./qwen_tts_avx512vnni            -j4
echo
echo "[C] Batched matmat (--matmat-bench): weight read ONCE for B streams (server throughput)."
echo "    SPEEDUP>1 = batching beats B separate matvecs. int8/q4 now use VNNI (was f32-dequant)."
./qwen_tts_avx512vnni --matmat-bench 2>&1 | grep -iE 'bf16|int8|int4|SPEEDUP' | head -9
echo
echo "[D] TTFA (streaming, ms to first audio chunk):"
ttfa "VNNI bf16" ./qwen_tts_avx512vnni
ttfa "VNNI int8" ./qwen_tts_avx512vnni --int8
ttfa "VNNI int4" ./qwen_tts_avx512vnni --int4

if [ "${BATCH:-0}" = "1" ]; then
  echo
  echo "[E] e2e server request-batching (matched: N concurrent to a batch-size-N server):"
  REQ="{\"text\":\"$TXT\",\"seed\":42,\"speaker\":\"ryan\",\"language\":\"English\"}"
  for N in 1 2 4; do
    pkill -9 -x qwen_tts 2>/dev/null; sleep 1
    ./qwen_tts_avx512vnni -d "$MODEL" --serve 8010 --batch-size "$N" --int8 --seed 42 >/tmp/sv.log 2>&1 &
    for i in $(seq 1 40); do grep -qi listening /tmp/sv.log && break; sleep 1; done
    t0=$(date +%s.%N); for j in $(seq 1 "$N"); do timeout 120 curl -s http://localhost:8010/v1/tts -d "$REQ" -o /tmp/e$j.wav & done; wait; t1=$(date +%s.%N)
    ok=0; for j in $(seq 1 "$N"); do [ -s /tmp/e$j.wav ] && ok=$((ok+1)); done
    printf "  batch=%s + %s concurrent int8:  wall=%.1fs  ok=%s/%s\n" "$N" "$N" "$(echo "$t1-$t0"|bc)" "$ok" "$N"
    pkill -9 -x qwen_tts 2>/dev/null
  done
fi

echo "================================================================"
echo "Read [A]: if 'scalar bf16 -j1' RTF >> 'VNNI int8 -j1', the kernel stack works"
echo "(weak -j4 scaling on a small VM is the box, not the code). On x86 single-stream"
echo "int8-VNNI is the winner; int4 has extra nibble-unpack so it trails int8 (unlike"
echo "M1 where int4-SDOT wins). Batching [C] is where int8/q4 VNNI pays. Paste the table."
