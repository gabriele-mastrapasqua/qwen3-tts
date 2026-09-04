#!/usr/bin/env bash
# tools/source_fingerprint.sh — the real identity of the source tree, computed where git is.
#
#   <commit>:clean                      tracked tree, nothing modified
#   <commit>-dirty:<tree12>             tree12 = sha256 (12 hex) of `git diff HEAD` on the
#                                       engine sources + every modified/untracked source
#                                       file's own sha256, so two dirty trees with different
#                                       edits never share a fingerprint
#   (no git)  -> the .source_fingerprint file shipped with the sync, or "unknown"
#
# The Makefile embeds this in the binary (qwen_build_id.h -> `--caps` "src=" and the
# dispatch map), so a rented box without .git reports the fingerprint FROM THE BINARY and
# no benchmark needs a commit declared by hand (QWEN_SOURCE_COMMIT is a fallback only).
set -u
cd "$(dirname "$0")/.." || exit 1
sha12() { if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -c1-12; else shasum -a 256 | cut -c1-12; fi; }
shaf()  { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
SCOPE=( '*.c' '*.h' 'Makefile' 'tests' 'tools' 'configs' 'third_party' 'vendor' )
if git rev-parse HEAD >/dev/null 2>&1; then
    c=$(git rev-parse --short HEAD)
    changed=$(git status --porcelain --untracked-files=all -- "${SCOPE[@]}" 2>/dev/null)
    if [ -n "$changed" ]; then
        t=$( { git diff HEAD -- "${SCOPE[@]}"
               printf '%s\n' "$changed" | awk '{print $NF}' | sort | while IFS= read -r f; do
                   [ -f "$f" ] && printf '%s %s\n' "$f" "$(shaf "$f")"
               done; } | sha12 )
        fp="$c-dirty:$t"
    else
        fp="$c:clean"
    fi
    printf '%s\n' "$fp" > .source_fingerprint
else
    fp=$(cat .source_fingerprint 2>/dev/null | tr -d '[:space:]')
    fp=${fp:-unknown}
fi
printf '%s\n' "$fp"
