#!/usr/bin/env bash
set -eu

echo "── setup_m2: bootstrapping $(uname -srm) ──"
command -v sysctl >/dev/null && sysctl -n machdep.cpu.brand_string 2>/dev/null

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "── installing Command Line Tools (headless) ──"
    sudo touch /Library/Developer/CommandLineTools 2>/dev/null || true
    FLAG=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
    sudo touch "$FLAG"
    PROD=$(softwareupdate -l 2>/dev/null \
            | grep -E 'Command Line Tools' | tail -1 \
            | sed -E 's/^[^C]*Label: *//; s/^\* *Label: *//' | tr -d '\n')
    if [ -n "${PROD:-}" ]; then
      echo "   installing: $PROD"
      sudo softwareupdate -i "$PROD" --verbose || true
    fi
    sudo rm -f "$FLAG" || true
    if ! xcode-select -p >/dev/null 2>&1; then
      echo "!! CLT still missing. Fall back to: xcode-select --install (may need a GUI/VNC session),"
      echo "   or use PATH A (scp the prebuilt binary + SKIP_BUILD=1 ./setup_m2.sh)."
      exit 1
    fi
  fi
  echo "   CLT: $(xcode-select -p)"
fi

if [ "${SKIP_MODELS:-0}" != "1" ]; then
  [ -d qwen3-tts-0.6b ] || { echo "── download 0.6B CustomVoice ──"; ./download_model.sh --model small; }
  [ -d qwen3-tts-1.7b ] || { echo "── download 1.7B CustomVoice ──"; ./download_model.sh --model large; }
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "── make metal CC=clang ──"
  make metal CC=clang
  echo "── native build done. --caps: ──"
  ./qwen_tts --caps 2>&1 | grep -iE "note:|lever" || true
else
  echo "── SKIP_BUILD=1: expecting an scp'd ./qwen_tts binary ──"
  [ -x ./qwen_tts ] && ./qwen_tts --caps 2>&1 | grep -iE "note:|lever" || echo "!! ./qwen_tts not found — scp it here first."
fi

echo ""
echo "✅ setup done. Now run:  ./bench_m2.sh"
