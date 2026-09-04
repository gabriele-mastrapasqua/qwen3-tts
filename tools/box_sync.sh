#!/usr/bin/env bash
# tools/box_sync.sh user@host [remote_dir]  — ship the tree to a rented box WITH its fingerprint.
#
# Computes tools/source_fingerprint.sh here (where git is), writes .source_fingerprint, and
# tars sources + tests + tools + configs + third_party sources to the box.  Excludes every
# build product (a Mac-built libingot.a or .o would poison an x86 link) and the models.
# rsync 3.4.x on the receiver rejects long exclude lists; tar over ssh does not.
set -u
[ $# -ge 1 ] || { echo "usage: $0 user@host [remote_dir=~/qwen-a1/src] [ssh_key=~/.ssh/id_ed25519]"; exit 2; }
HOST=$1; DEST=${2:-'~/qwen-a1/src'}; KEY=${3:-$HOME/.ssh/id_ed25519}
cd "$(dirname "$0")/.." || exit 1
FP=$(bash tools/source_fingerprint.sh)
echo "fingerprint: $FP"
tar czf - --exclude='*.o' --exclude='*.o.tmp' --exclude='*.d' --exclude='*.a' --exclude='*.wav' \
    --exclude='__pycache__' --exclude='tools/para/.venv' --exclude='tests/golden' \
    --exclude='qwen_build_id.h' --exclude='profiles' \
    ./*.c ./*.h Makefile download_model.sh bench.sh .source_fingerprint tests tools configs vendor third_party \
  | ssh -i "$KEY" "$HOST" "mkdir -p $DEST && tar xzf - -C $DEST 2>/dev/null && cd $DEST && rm -f third_party/ingot/libingot.a qwen_build_id.h && echo synced: \$(cat .source_fingerprint)"
