#!/usr/bin/env bash
set -euo pipefail

bin=${1:-./qwen_tts}
if [[ ! -x "$bin" ]]; then
    echo "FAIL: executable not found: $bin" >&2
    exit 1
fi

expect_pool() {
    local requested=$1 expected=$2 output
    if ! output=$(QWEN_SD_POOL="$requested" "$bin" --dispatch-map 2>&1); then
        echo "FAIL: QWEN_SD_POOL=$requested unexpectedly failed" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
    if ! printf '%s\n' "$output" | grep -Eq "decoder\.pool.*${expected}"; then
        echo "FAIL: QWEN_SD_POOL=$requested did not resolve to $expected" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
}

for alias in engine qwen q 1; do expect_pool "$alias" engine; done
for alias in private 0; do expect_pool "$alias" private; done

effective=$(QWEN_SD_POOL=engine "$bin" --effective-config 2>&1)
if ! printf '%s\n' "$effective" | grep -Eq 'QWEN_SD_POOL.*requested=engine resolved=engine'; then
    echo "FAIL: effective-config did not report requested/resolved pool" >&2
    printf '%s\n' "$effective" >&2
    exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
if QWEN_SD_POOL=typo "$bin" --dispatch-map >"$tmp" 2>&1; then
    echo "FAIL: invalid QWEN_SD_POOL was accepted" >&2
    cat "$tmp" >&2
    exit 1
fi
if ! grep -Eq 'FATAL.*QWEN_SD_POOL.*invalid' "$tmp"; then
    echo "FAIL: invalid QWEN_SD_POOL did not produce a clear fatal error" >&2
    cat "$tmp" >&2
    exit 1
fi

echo "PASS: QWEN_SD_POOL aliases, resolved reporting, and fail-fast validation"
