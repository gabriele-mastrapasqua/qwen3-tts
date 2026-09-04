#!/usr/bin/env bash
# tools/cpu_check.sh — the 10-20 second preflight every CPU optimisation session starts with.
#
#   make cpu-check                         # defaults
#   make cpu-check CPU_PROFILE=<id>        # also resolve a configs/perf profile + forbidden env
#   make cpu-check CPU_MODEL=<model dir>   # also fingerprint the model (config + file sizes)
#
# It does not need a model and does not run the server.  It produces ONE artifact
# directory, profiles/<date>_<host>_<binsha8>/, with:
#
#   manifest.json   provenance (binary sha, build tag, commit, dirty-source hash, compiler,
#                   host, env, model fingerprint, resolved dispatch) — tools/profile_check.py
#   hardware.json   tools/box_info.sh (+ measured bandwidth when tests/membw.c is built)
#   hardware.txt    the readable box report
#   build.txt       `make info` + compiler version
#   caps.txt        ./qwen_tts --caps
#   selftest_native.txt / selftest_fallback.txt
#   dispatch.txt / dispatch.json   ./qwen_tts --dispatch-map (RESOLVED flags, not raw env)
#   dispatch_gate.txt              expected-vs-observed for this ISA class
#   env.txt         every QWEN_* / OPENBLAS_* / OMP_* variable present
#   gate.txt        the PASS/FAIL/WARN/SKIP lines below
#
# and exits non-zero on any FAIL.  profiles/LATEST points at the newest run.
#
# The gates encode the measurement traps of plan §3 / docs/ENGINEERING-METHOD.md: a stale
# binary, a dirty tree without a fingerprint, a fallback dispatch nobody asked for, a box
# with SMT/governor/quota problems, or leftover harness processes each make every number
# that follows describe a different machine.
set -u
cd "$(dirname "$0")/.." || exit 1

BIN=${BIN:-./qwen_tts}
PROFILES_DIR=${PROFILES_DIR:-profiles}
MEMBW_BIN=${MEMBW_BIN:-}
CPU_PROFILE=${CPU_PROFILE:-}
CPU_MODEL=${CPU_MODEL:-}
CPU_CHECK_STRICT=${CPU_CHECK_STRICT:-0}     # 1 = hardware gates (SMT/governor/quota) FAIL instead of WARN
SKIP_SELFTEST=${SKIP_SELFTEST:-0}

NPASS=0; NFAIL=0; NWARN=0; NSKIP=0
GATES=""
gate() {   # gate STATUS name detail
    local st="$1" name="$2" detail="${3:-}"
    case "$st" in PASS) NPASS=$((NPASS+1));; FAIL) NFAIL=$((NFAIL+1));; WARN) NWARN=$((NWARN+1));; SKIP) NSKIP=$((NSKIP+1));; INFO) ;; esac
    local line
    line=$(printf '[%s] %-36s %s' "$st" "$name" "$detail")
    echo "$line"
    GATES="$GATES$line
"
}
have() { command -v "$1" >/dev/null 2>&1; }
sha256() { if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

echo "CPU CHECK"
echo "========="

# ── 0. process-clean, before anything else touches the box ─────────────────────────
# Kill harnesses, not just servers, and prove zero (plan §3.1): a surviving wave/load_test
# from a previous campaign silently ran concurrently with the next one once.
# (pgrep -c is Linux-only; count lines instead so macOS cannot silently report 0)
SURV=$(pgrep -f '(^|/)qwen_tts( |$)|serve_parallel_wave|load_test\.py|serve_soak\.py' 2>/dev/null | wc -l | tr -d ' ')
SURV=${SURV:-0}
if [ "$SURV" = "0" ]; then gate PASS "process-clean" "surviving=0"
else gate FAIL "process-clean" "surviving=$SURV  (pgrep -fl 'qwen_tts|serve_parallel_wave|load_test.py' and kill them)"; fi

# ── 1. binary ───────────────────────────────────────────────────────────────────────
[ -x "$BIN" ] || { gate FAIL "binary present" "$BIN missing: make blas"; echo "$GATES"; exit 1; }
BSHA=$(sha256 "$BIN")
CAPS=$("$BIN" --caps 2>&1)
BTAG=$(printf '%s\n' "$CAPS" | sed -n 's/^  build: *\([^ ]*\).*/\1/p' | head -1)
BSIMD=$(printf '%s\n' "$CAPS" | sed -n 's/.*SIMD=\([^ ]*\).*/\1/p' | head -1)
gate PASS "binary SHA recorded" "${BSHA:0:16}…  build=$BTAG simd=$BSIMD"

# ── 2. source vs binary ─────────────────────────────────────────────────────────────
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "")
if [ -n "$COMMIT" ]; then
    DIRTY=$(git status --porcelain 2>/dev/null | grep -q . && echo yes || echo no)
    TREE_TAG="$COMMIT$([ "$DIRTY" = yes ] && echo -dirty)"
    if [ "$BTAG" = "$TREE_TAG" ]; then
        gate PASS "binary matches tree" "$TREE_TAG"
    elif [ "${BTAG%-dirty}" = "$COMMIT" ]; then
        gate WARN "binary matches tree" "binary=$BTAG tree=$TREE_TAG (dirty state changed since the build: rebuild before measuring)"
    else
        gate FAIL "binary matches tree" "binary=$BTAG tree=$TREE_TAG  -> STALE BINARY, make blas"
    fi
    if [ "$DIRTY" = yes ]; then
        gate WARN "dirty tree" "uncommitted edits: numbers that leave the repo need dirty=no"
    fi
else
    DIRTY="unknown"; TREE_TAG="${QWEN_SOURCE_COMMIT:-UNKNOWN}"
    gate WARN "binary matches tree" "no git here; QWEN_SOURCE_COMMIT=${QWEN_SOURCE_COMMIT:-unset}"
fi

# ── 3. artifact directory ───────────────────────────────────────────────────────────
HOSTSLUG=$(hostname -s 2>/dev/null | tr -cd 'a-zA-Z0-9-' | cut -c1-24); HOSTSLUG=${HOSTSLUG:-box}
OUT="$PROFILES_DIR/$(date +%Y-%m-%d_%H%M%S)_${HOSTSLUG}_${BSHA:0:8}"
mkdir -p "$OUT" || exit 1
printf '%s\n' "$CAPS" > "$OUT/caps.txt"
env | grep -E '^(QWEN_|OPENBLAS_|OMP_|GOMP_|KMP_|MKL_)' | sort > "$OUT/env.txt" || true

# ── 4. hardware ─────────────────────────────────────────────────────────────────────
if [ -r tools/box_info.sh ]; then
    if [ -n "$MEMBW_BIN" ] && [ -x "$MEMBW_BIN" ]; then
        MEMBW_BIN="$MEMBW_BIN" bash tools/box_info.sh --out "$OUT/hardware.json" > "$OUT/hardware.txt" 2>&1
        BWNOTE="bandwidth measured"
    else
        bash tools/box_info.sh --out "$OUT/hardware.json" > "$OUT/hardware.txt" 2>&1
        BWNOTE="bandwidth NOT measured (build tests/membw.c: make membw)"
    fi
    if [ -s "$OUT/hardware.json" ]; then
        gate PASS "hardware fingerprint" "$OUT/hardware.json  ($BWNOTE)"
        # the three invalidating conditions box_info.sh already computes
        python3 - "$OUT/hardware.json" "$CPU_CHECK_STRICT" <<'PY' > "$OUT/hw_gates.txt"
import json, sys
d = json.load(open(sys.argv[1])); strict = sys.argv[2] == "1"
g = d.get("gates") or {}
for k, label in (("smt_off", "SMT off"), ("governor_performance", "governor performance"), ("no_cgroup_quota", "no cgroup quota")):
    v = g.get(k, "n/a")
    st = "PASS" if v == "PASS" else ("FAIL" if strict else "WARN")
    print(f"{st}\t{label}\t{v}")
PY
        while IFS=$'\t' read -r st label v; do
            [ -n "$st" ] && gate "$st" "hw: $label" "$v$([ "$st" != PASS ] && echo '  (cloud slice? CPU_CHECK_STRICT=1 to fail)')"
        done < "$OUT/hw_gates.txt"
    else
        gate WARN "hardware fingerprint" "box_info.sh produced no JSON (see $OUT/hardware.txt)"
    fi
else
    gate SKIP "hardware fingerprint" "tools/box_info.sh missing"
fi

# ── 5. build / compiler ─────────────────────────────────────────────────────────────
{ make -s info 2>/dev/null; echo; CC_=${CC:-cc}; echo "cc: $CC_"; $CC_ --version 2>/dev/null | head -2; } > "$OUT/build.txt" 2>&1
gate PASS "compiler + flags recorded" "$OUT/build.txt"

# ── 6. flag registry ────────────────────────────────────────────────────────────────
if python3 tools/check_flag_registry.py > "$OUT/flag_registry.txt" 2>&1; then
    gate PASS "flag registry" "$(tail -1 "$OUT/flag_registry.txt")"
else
    gate FAIL "flag registry" "$(grep -m1 FAIL "$OUT/flag_registry.txt")"
fi

# ── 7. kernel correctness, native and fallback ──────────────────────────────────────
if [ "$SKIP_SELFTEST" = "1" ]; then
    gate SKIP "self-test" "SKIP_SELFTEST=1"
else
    if "$BIN" --self-test > "$OUT/selftest_native.txt" 2>&1; then gate PASS "self-test native" "PASS"
    else gate FAIL "self-test native" "see $OUT/selftest_native.txt"; fi
    if QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 QWEN_NO_AMX=1 "$BIN" --self-test > "$OUT/selftest_fallback.txt" 2>&1; then
        gate PASS "self-test fallback" "QWEN_NO_SDOT=1 QWEN_NO_VNNI=1 QWEN_NO_AMX=1"
    else gate FAIL "self-test fallback" "see $OUT/selftest_fallback.txt"; fi
fi

# ── 8. resolved dispatch map ────────────────────────────────────────────────────────
if QWEN_DISPATCH_JSON="$OUT/dispatch.json" "$BIN" --dispatch-map > "$OUT/dispatch.txt" 2>&1 && [ -s "$OUT/dispatch.json" ]; then
    ISA=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["isa_class"])' "$OUT/dispatch.json")
    gate PASS "resolved dispatch map" "isa_class=$ISA  $OUT/dispatch.txt"
    python3 tools/dispatch_gate.py --coverage "$OUT/dispatch.json" > "$OUT/coverage.txt" 2>&1 \
        && gate INFO "feature coverage" "$(grep -m1 'resolved by the map' "$OUT/coverage.txt" | sed 's/^ *//')"
    if python3 tools/dispatch_gate.py "$OUT/dispatch.json" > "$OUT/dispatch_gate.txt" 2>&1; then
        gate PASS "expected-vs-observed dispatch" "no SUSPICIOUS / MISMATCH for $ISA"
    else
        gate FAIL "expected-vs-observed dispatch" "$(grep -c -E 'SUSPICIOUS|MISMATCH' "$OUT/dispatch_gate.txt") finding(s), see below"
    fi
else
    gate FAIL "resolved dispatch map" "--dispatch-map failed (binary without it? rebuild)"
fi

# ── 9. serving profile (optional) ───────────────────────────────────────────────────
if [ -n "$CPU_PROFILE" ]; then
    if SENV=$(python3 tools/perf_profile.py server-env "$CPU_PROFILE" 2>"$OUT/profile_err.txt"); then
        gate PASS "profile resolves" "$CPU_PROFILE -> $SENV"
        FORB_PRESENT=""
        for V in $(python3 tools/perf_profile.py forbidden-env "$CPU_PROFILE" 2>/dev/null); do
            eval "P=\${$V+set}"; [ "${P:-}" = set ] && FORB_PRESENT="$FORB_PRESENT $V"
        done
        if [ -z "$FORB_PRESENT" ]; then gate PASS "forbidden env absent" "none present"
        else gate FAIL "forbidden env absent" "present:$FORB_PRESENT (the profile declares they must be ABSENT)"; fi
    else
        gate FAIL "profile resolves" "$(head -1 "$OUT/profile_err.txt")"
    fi
else
    gate SKIP "profile resolves" "CPU_PROFILE= not given"
fi

# ── 10. model fingerprint (optional) ────────────────────────────────────────────────
if [ -n "$CPU_MODEL" ]; then
    if [ -d "$CPU_MODEL" ]; then gate PASS "model fingerprint" "$CPU_MODEL (config + file sizes)"
    else gate FAIL "model fingerprint" "$CPU_MODEL is not a directory"; fi
else
    gate SKIP "model fingerprint" "CPU_MODEL= not given"
fi

# ── 11. perf availability (informational) ───────────────────────────────────────────
if have perf; then gate PASS "perf available" "$(perf --version 2>/dev/null | head -1)"
else gate SKIP "perf available" "not installed (Linux: linux-tools); profile-cpu will skip the system profile"; fi

# ── 12. manifest ────────────────────────────────────────────────────────────────────
python3 tools/profile_check.py --fingerprint --bin "$BIN" ${CPU_MODEL:+--model "$CPU_MODEL"} > "$OUT/manifest.json" 2>"$OUT/manifest_err.txt"
if [ -s "$OUT/manifest.json" ]; then
    python3 - "$OUT/manifest.json" "$OUT" "$CPU_PROFILE" "$NFAIL" "$NWARN" <<'PY'
import json, sys
p, out, prof, nfail, nwarn = sys.argv[1:6]
d = json.load(open(p))
d["artifact_dir"] = out
d["tool"] = "tools/cpu_check.sh"
d["serving_profile"] = prof or None
d["gate_fail"] = int(nfail); d["gate_warn"] = int(nwarn)
try:
    d["hardware"] = json.load(open(out + "/hardware.json"))
except Exception:
    pass
json.dump(d, open(p, "w"), indent=1)
PY
    gate PASS "manifest written" "$OUT/manifest.json"
else
    gate FAIL "manifest written" "$(head -1 "$OUT/manifest_err.txt")"
fi

# ── summary ─────────────────────────────────────────────────────────────────────────
echo
if [ -s "$OUT/coverage.txt" ]; then
    echo "COVERAGE (what the map can and cannot say — full table in $OUT/coverage.txt)"; echo "--------"
    sed -n '1,2p' "$OUT/coverage.txt"; grep -E '^\s+TOTAL|resolved by the map' "$OUT/coverage.txt"; echo
fi
if [ -s "$OUT/dispatch_gate.txt" ]; then
    echo "DISPATCH"; echo "--------"
    sed 's/^/  /' "$OUT/dispatch_gate.txt"; echo
fi
if [ -s "$OUT/dispatch.txt" ]; then
    echo "RESOLVED FEATURES (full table in $OUT/dispatch.txt)"; echo "-----------------"
    awk '/^\[DISPATCH\]/{p=1} /^\[DISPATCH-GATE\]/{p=0} p' "$OUT/dispatch.txt" | sed -n '2,40p' | sed 's/^/ /'
    echo
fi
printf '%s' "$GATES" > "$OUT/gate.txt"
{
    echo; echo "PASS=$NPASS FAIL=$NFAIL WARN=$NWARN SKIP=$NSKIP"
    if [ "$NFAIL" = "0" ]; then echo "CPU CHECK VALID: YES"; else echo "CPU CHECK VALID: NO"; fi
} | tee -a "$OUT/gate.txt"
ln -sfn "$(basename "$OUT")" "$PROFILES_DIR/LATEST"
echo "artifacts: $OUT   (profiles/LATEST -> $(basename "$OUT"))"
[ "$NFAIL" = "0" ]
