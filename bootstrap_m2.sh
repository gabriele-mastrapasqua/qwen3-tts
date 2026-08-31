#!/usr/bin/env bash
set -eu

REPO_URL=${REPO_URL:-https://github.com/gabriele-mastrapasqua/qwen3-tts.git}
BRANCH=${BRANCH:-feat/gpu-backends}
WORKDIR=${WORKDIR:-$HOME/qwen-tts}

echo "════════════════════════════════════════════════════════════════════"
echo " bootstrap_m2 — fresh-box → native build → bench"
echo " host: $(uname -srm)"
command -v sysctl >/dev/null && echo " chip: $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
echo "════════════════════════════════════════════════════════════════════"

if ! xcode-select -p >/dev/null 2>&1 || ! /usr/bin/xcrun --find clang >/dev/null 2>&1; then
  echo "── [1/5] installing Command Line Tools (headless) ──"
  FLAG=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  sudo touch "$FLAG"
  PROD=$(softwareupdate -l 2>/dev/null | grep -E 'Label: *Command Line Tools' \
          | sed -E 's/.*Label: *//' | sort -V | tail -1)
  if [ -n "${PROD:-}" ]; then
    echo "   installing: $PROD"
    sudo softwareupdate -i "$PROD" --verbose || true
  fi
  sudo rm -f "$FLAG" || true
  xcode-select -p >/dev/null 2>&1 || sudo xcode-select --switch /Library/Developer/CommandLineTools 2>/dev/null || true
  if ! /usr/bin/xcrun --find clang >/dev/null 2>&1; then
    echo "!! CLT install did not complete non-interactively."
    echo "   Open the box's VNC/console once and run:  xcode-select --install"
    echo "   then re-run this script. (Many Scaleway images ship CLT/Xcode already.)"
    exit 1
  fi
fi
echo "   CLT ok: $(xcode-select -p)  ·  clang $(clang --version | head -1)"

if [ "${WITH_BREW:-0}" = "1" ] && ! command -v brew >/dev/null 2>&1; then
  echo "── installing Homebrew (optional) ──"
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || \
    echo "   (brew install skipped/failed — not required, continuing)"
fi

echo "── [2/5] fetch repo @ $BRANCH → $WORKDIR ──"
if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" fetch --depth 1 origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
echo "   at $(git rev-parse --short HEAD): $(git log -1 --pretty=%s)"

if [ "${SKIP_MODELS:-0}" != "1" ]; then
  echo "── [3/5] download models (curl from HF CDN) ──"
  chmod +x download_model.sh
  [ -d qwen3-tts-0.6b ] || ./download_model.sh --model small
  [ -d qwen3-tts-1.7b ] || ./download_model.sh --model large
fi

echo "── [4/5] make metal CC=clang (native → M2/M4 CPU i8mm/bf16 + Metal) ──"
make metal CC=clang
echo "── build ok. compiled caps: ──"
./qwen_tts --caps 2>&1 | grep -iE "runtime cpu|lever|note:" || true

if [ "${RUN_BENCH:-1}" = "1" ]; then
  echo "── [5/5] running bench_m2.sh (CPU + Metal, full RTF matrix) ──"
  chmod +x bench_m2.sh
  ./bench_m2.sh
  echo ""
  echo "✅ DONE. Send back:   cat $WORKDIR/bench_out/summary_*.txt"
else
  echo "✅ build done (RUN_BENCH=0). Bench with:  cd $WORKDIR && ./bench_m2.sh"
fi
