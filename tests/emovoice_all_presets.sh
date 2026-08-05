#!/bin/bash
# Build the dedicated emotional voice assets for ALL 9 preset voices, so `--emotion` on the 0.6B
# is zero-setup for every preset (not just ryan). 9 voices x 6 emotions x 4 KB = ~216 KB shipped.
#
# Each preset gets its donors in a language it speaks NATIVELY — the donor's delivery is what we
# capture, and a voice fighting the language makes a worse donor. See the native-preset-per-language
# rule in docs/emotion-THE-recipe.md.
#
#   bash tests/emovoice_all_presets.sh          # ~40 min: 48 donors on the 1.7B + extraction
#
# Donors are cached under samples/tests/emovoice_donors/<voice>/, so re-running is free.
set -u
cd "$(dirname "$0")/.."

# voice:language — ryan already ships (Italian); rebuild it too only if its assets are missing.
PRESETS="ryan:Italian vivian:Chinese uncle_fu:Chinese ono_anna:Japanese sohee:Korean serena:English aiden:English eric:English dylan:English"

# Fail fast and loudly: both models must be reachable for the WHOLE run. Checking per-voice (as the
# builder does) means a network volume that unmounts mid-run silently skips the remaining presets.
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
