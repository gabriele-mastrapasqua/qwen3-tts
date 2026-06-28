#!/usr/bin/env bash
# ============================================================================================
# download_assets.sh — fetch the large VALIDATED expressivity .expr weight-deltas.
#
# WHY: .expr files are 30-200MB each (presets/expr/ ~1.4GB) → NOT committed to git. They are
# hosted as release assets; this script fetches the wins on demand. The tiny steering vectors
# (presets/steer/**) ARE committed — no download needed for those.
#
# Usage:   bash download_assets.sh            # download any missing win .expr
#          bash download_assets.sh --verify   # only verify sha256 of present files
#          BASE_URL=<url> bash download_assets.sh   # override host
# ============================================================================================
set -uo pipefail
cd "$(dirname "$0")"
DEST=presets/expr
# HF repo hosting the win .expr (filename appended). Override with BASE_URL=... if you mirror elsewhere.
BASE_URL="${BASE_URL:-https://huggingface.co/gabrione/qwen3-tts-italian-expr/resolve/main}"
mkdir -p "$DEST"

# filename  sha256  (the 9 validated wins — see presets/expr/MANIFEST.md)
read -r -d '' ASSETS <<'EOF'
italian_csp_topk6.expr            9315f80a9181b18db6148574b92946ec2fa0af3234a3c2e72cdb5bd066cc58d9
italian_csp_topk4.expr            6a65f031fe06c55893877f97fb5f11357434242c2b3f1c862ec4ff69f52ed211
german_csp_k6.expr                f77d471da62fff068222da7f9961b0ef037c20c62ff33ee0e59919bfea1533fe
french_csp_k6.expr                eff743ea6cae4ab2d26931b77cff4ce1ab1621978f45eda00af1f34197470ce2
italian_l1626_dense.expr          dc86f9de302d6afc888dfb10e9a5f3134363a4101cfcef577a9cc8895239f671
italian_l1626_r32.expr            927e8bec191e00f645136cc16361859b75f28efdf6982653d1bd44eb52005fa2
italian_l1626_r64.expr            9efb9143574cabca9598570e3fb4145245ffb9ccb20b22ab526fa215c97e2930
italian_multi_l1626_dense.expr    9445275fd058c18a0f48d82bb465ff820a19bf99a3d508bd4c1da8cb555c0e3a
italian_multitag_l1626_dense.expr 6e986b6e7d8f0ae80bcc1d2c4190f511eaa24e6ebebf69a1e19e6f1bbc913528
EOF

VERIFY_ONLY=0; [ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
sha(){ shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }

echo "$ASSETS" | while read -r name want; do
  [ -z "$name" ] && continue
  f="$DEST/$name"
  if [ -f "$f" ]; then
    got=$(sha "$f")
    if [ "$got" = "$want" ]; then echo "OK    $name"; else echo "BAD   $name (sha mismatch — re-download)"; [ $VERIFY_ONLY = 1 ] || rm -f "$f"; fi
  fi
  [ $VERIFY_ONLY = 1 ] && continue
  if [ ! -f "$f" ]; then
    echo "GET   $name  <- $BASE_URL/$name"
    if curl -fL --retry 3 -o "$f.part" "$BASE_URL/$name"; then
      mv "$f.part" "$f"
      got=$(sha "$f"); [ "$got" = "$want" ] && echo "  verified" || echo "  WARNING sha mismatch after download"
    else
      echo "  FAILED (set BASE_URL to the real host; see presets/expr/MANIFEST.md)"; rm -f "$f.part"
    fi
  fi
done
echo "done. (steering vectors in presets/steer/** are committed — no download needed.)"
