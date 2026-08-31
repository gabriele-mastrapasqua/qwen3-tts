#!/bin/bash
set -u
cd "$(dirname "$0")/.."

PRESETS="ryan:Italian vivian:Chinese uncle_fu:Chinese ono_anna:Japanese sohee:Korean serena:English aiden:English eric:English dylan:English"

for m in qwen3-tts-1.7b qwen3-tts-0.6b-base; do
  if [ ! -d "$m" ] || [ ! -f "$m/config.json" ]; then
    echo "Error: '$m' is not reachable (missing, or a symlink to an unmounted volume)."
    echo "  This run needs BOTH models for its whole duration: the 1.7B renders the emotional"
    echo "  donors, the 0.6B Base extracts the 4 KB voices. Mount/download it and re-run —"
    echo "  already-built voices are skipped, so resuming is free."
    exit 1
  fi
done

for spec in $PRESETS; do
  v="${spec%%:*}"; lang="${spec##*:}"
  if [ -f "presets/emovoice/${v}_ang.bin" ] && [ -f "presets/emovoice/${v}_surprise.bin" ]; then
    echo "=== $v — already built, skipping ==="
    continue
  fi
  echo
  echo "=========== $v ($lang) ==========="
  VOICE="$v" TTS_LANG="$lang" bash tests/emovoice_build.sh
done

echo
echo "=== shipped assets ==="
ls presets/emovoice/*.bin | wc -l | xargs echo "  files:"
du -sh presets/emovoice | awk '{print "  total: " $1}'
