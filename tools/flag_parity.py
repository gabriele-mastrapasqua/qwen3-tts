#!/usr/bin/env python3
"""flag_parity.py — per-backend applicability of every runtime flag, derived from the code.

PARITY-2 asks a question docs cannot answer honestly: for each flag, is it IMPLEMENTED,
SELECTABLE and ACTUALLY EFFECTIVE on each backend?  A flag that parses and does nothing on a
backend is a parity defect (QWEN_NO_SIMD_QUANT was exactly that on ARM: it existed, it looked
like a feature, and the NEON path never consulted it).

Ground truth is the read site, not the documentation.  Every QWEN_* literal in the sources is
an environment variable name; this walks the preprocessor conditionals around each occurrence
and reports which backend families can reach it.  Then it compares that with the ISA column the
documentation claims, and flags the disagreements.

    tools/flag_parity.py                 # human table + verdicts
    tools/flag_parity.py --json out.json # machine-readable matrix
    tools/flag_parity.py --check         # non-zero exit if a flag's claim disagrees with code
"""
import argparse, glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# One backend family per column.  A guard token maps to the families that can compile it.
FAMILIES = ["common", "arm", "avx2", "avx512f", "vnni", "amx", "apple", "gpu"]
X86 = ["avx2", "avx512f", "vnni", "amx"]
TOKEN_FAMILIES = {
    "__ARM_NEON": ["arm", "apple"], "__aarch64__": ["arm", "apple"],
    "__ARM_FEATURE_DOTPROD": ["arm", "apple"], "__ARM_FEATURE_MATMUL_INT8": ["arm"],
    "__ARM_FEATURE_SVE": ["arm"], "__ARM_FEATURE_BF16_VECTOR_ARITHMETIC": ["arm"],
    "QWEN_HAVE_KLEIDI": ["arm"], "QWEN_KLEIDI": ["arm"],
    "__AVX2__": X86, "__FMA__": X86, "__x86_64__": X86, "_M_X64": X86,
    "__AVX512F__": ["avx512f", "vnni", "amx"], "__AVX512BW__": ["avx512f", "vnni", "amx"],
    "__AVX512VL__": ["avx512f", "vnni", "amx"], "__AVX512DQ__": ["vnni", "amx"],
    "__AVX512VNNI__": ["vnni", "amx"], "__AVX512BF16__": ["vnni", "amx"],
    "__AMX_INT8__": ["amx"], "__AMX_TILE__": ["amx"], "__AMX_BF16__": ["amx"],
    "__APPLE__": ["apple"],
    "QWEN_HAVE_CUDA": ["gpu"], "QWEN_HAVE_METAL": ["gpu"], "QWEN_USE_METAL": ["gpu"],
}
IGNORE_TOKENS = {"defined", "USE_BLAS", "USE_OPENBLAS", "QWEN_ASAN", "__GNUC__", "__clang__",
                 "QWEN_HAVE_MADVISE", "QWEN_SIMD_PROFILE", "_WIN32", "QWEN_USE_PTHREADS",
                 "ACCELERATE_NEW_LAPACK", "QWEN_MAYBE_UNUSED"}

COND = re.compile(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$')
FLAG = re.compile(r'"(QWEN_[A-Z0-9_]+)"')


def families_of(cond_stack):
    """Which families can reach a site guarded by this stack of conditions."""
    reach = set(FAMILIES)
    for cond, negated in cond_stack:
        toks = [t for t in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', cond)
                if t not in IGNORE_TOKENS and t in TOKEN_FAMILIES]
        if not toks:
            continue
        allowed = set()
        for t in toks:
            allowed |= set(TOKEN_FAMILIES[t])
        if negated:
            # !defined(X): every family EXCEPT the ones the token names, but a family that has
            # other ways in is not removed by a single negated guard, so only narrow when the
            # token is family-exclusive.
            reach -= allowed if len(allowed) < len(FAMILIES) else set()
        else:
            reach &= allowed | {"common"} if "common" in reach and not toks else reach & allowed
    return reach


def scan():
    """Guard reach of every flag, following ONE call hop.

    The read site alone is the wrong signal: getenv usually lives in a small unguarded helper
    (quant_simd_off() reads QWEN_NO_SIMD_QUANT with no #if around it) while the ISA guard sits
    around the code that CONSULTS that helper.  So: find the function enclosing each read, then
    union the guard reach of every call site of that function.  A reader called from anywhere
    unguarded keeps full reach, which is the conservative answer.
    """
    FUNC = re.compile(r'^[A-Za-z_][A-Za-z0-9_ \*]*\b([a-z_][a-z0-9_]*)\s*\(')
    # qwen_tts_dispatch.c REPORTS flags (row(..., "QWEN_X", ...)) and qwen_flag_scope.h lists
    # them; neither implements an effect, and counting them handed every flag full reach.
    SKIP = {"qwen_tts_dispatch.c", "qwen_flag_scope.h"}
    files = [f for p in ("*.c", "*.h", "*.cu", "*.m") for f in glob.glob(os.path.join(ROOT, p))
             if os.path.basename(f) not in SKIP]
    read_in = {}        # flag -> set of enclosing function names
    flag_files = {}
    call_reach = {}     # function -> set of families over all its call sites
    direct = {}         # flag -> reach at the read site itself

    parsed = []
    for path in sorted(files):
        try:
            lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        except OSError:
            continue
        stack, fn, in_registry = [], None, False
        rows = []
        for ln in lines:
            # the registry array lists every flag name at file scope; it is a declaration,
            # not a read site, and counting it gave every flag full reach
            if "g_qwen_reported_flags[]" in ln:
                in_registry = True
            elif in_registry and ln.startswith("};"):
                in_registry = False
            if in_registry:
                continue
            m = COND.match(ln)
            if m:
                kind, rest = m.group(1), m.group(2)
                if kind in ("if", "ifdef", "ifndef"):
                    stack.append((rest, kind == "ifndef" or rest.strip().startswith("!")))
                elif kind == "elif":
                    if stack: stack[-1] = (rest, False)
                elif kind == "else":
                    if stack:
                        c, neg = stack[-1]; stack[-1] = (c, not neg)
                elif kind == "endif":
                    if stack: stack.pop()
                continue
            fm = FUNC.match(ln)
            if fm and "(" in ln and not ln.rstrip().endswith(";"):
                fn = fm.group(1)
            rows.append((ln, list(stack), fn))
        parsed.append((os.path.basename(path), rows))

    for base, rows in parsed:
        for ln, stack, fn in rows:
            for name in FLAG.findall(ln):
                read_in.setdefault(name, set()).add(fn)
                flag_files.setdefault(name, set()).add(base)
                d = direct.setdefault(name, set())
                d |= families_of(stack)

    readers = {f for fns in read_in.values() for f in fns if f}
    for base, rows in parsed:
        for ln, stack, fn in rows:
            if COND.match(ln):
                continue
            for r in readers:
                if fn != r and re.search(r'\b' + re.escape(r) + r'\s*\(', ln):
                    call_reach.setdefault(r, set()).update(families_of(stack))

    sites = {}
    for name, fns in read_in.items():
        # The guard AROUND THE READ already limits the flag; the call hop can only narrow it
        # further, never widen it.  Unioning call sites first was wrong: QWEN_APPLE_MMLA is read
        # inside #if defined(__APPLE__) within qwen_mm_use(), which is called from everywhere,
        # and the union handed it back full reach.
        via_calls = set()
        for fn in fns:
            if not fn:
                continue          # file-scope literal: a declaration, not a read
            via_calls |= call_reach.get(fn, set(FAMILIES))
        reach = direct.get(name, set(FAMILIES)) & (via_calls or set(FAMILIES))
        if not reach:
            reach = direct.get(name, set(FAMILIES))
        sites[name] = {"families": reach or set(FAMILIES), "files": flag_files.get(name, set())}
    return sites


DOC = os.path.join(ROOT, "docs", "feature-flags.md")
DOC_ROW = re.compile(r'^\|\s*(`QWEN_[^|]*?)\|\s*([^|]*?)\s*\|')


def documented_isa():
    claims = {}
    if not os.path.exists(DOC):
        return claims
    for ln in open(DOC, encoding="utf-8"):
        m = DOC_ROW.match(ln)
        if not m:
            continue
        isa = m.group(2).strip().lower()
        for name in re.findall(r'`(QWEN_[A-Z0-9_]+)`', m.group(1)):
            claims.setdefault(name, isa)
    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-c", metavar="PATH",
                    help="write the scope table the engine embeds for --effective-config")
    a = ap.parse_args()

    sites, claims = scan(), documented_isa()
    rows, problems = [], []
    for name in sorted(sites):
        fam = sites[name]["families"]
        arm_ok = bool(fam & {"arm"})
        x86_ok = bool(fam & set(X86))
        apple_ok = "apple" in fam
        scope = ("all" if arm_ok and x86_ok else
                 "arm" if arm_ok else "x86" if x86_ok else
                 "apple" if apple_ok else "gpu" if "gpu" in fam else "none")
        claim = claims.get(name, "")
        verdict = "ok"
        if claim:
            c = claim.split("/")[0].strip()
            claims_all = c in ("all", "any", "common", "cpu", "-", "")
            if claims_all and scope in ("arm", "x86"):
                verdict = "CLAIMS-ALL-BUT-%s-ONLY" % scope.upper()
            elif c.startswith("arm") and scope == "x86":
                verdict = "CLAIMS-ARM-BUT-X86-ONLY"
            elif c.startswith("x86") and scope == "arm":
                verdict = "CLAIMS-X86-BUT-ARM-ONLY"
        rows.append({"flag": name, "scope": scope, "documented_isa": claim or None,
                     "families": sorted(fam), "files": sorted(sites[name]["files"]),
                     "verdict": verdict})
        if verdict != "ok":
            problems.append(rows[-1])

    if a.emit_c:
        bits = {"arm": 1, "avx2": 2, "avx512f": 4, "vnni": 8, "amx": 16, "apple": 32, "gpu": 64}
        with open(a.emit_c, "w") as f:
            f.write("/* Generated by tools/flag_parity.py -- do not edit.\n"
                    " * Which backend families can reach each flag's effect, derived from the\n"
                    " * preprocessor guards around its read site and one call hop. */\n"
                    "#ifndef QWEN_FLAG_SCOPE_H\n#define QWEN_FLAG_SCOPE_H\n"
                    "#define QWEN_FSCOPE_ARM   1u\n#define QWEN_FSCOPE_AVX2  2u\n"
                    "#define QWEN_FSCOPE_AVX512F 4u\n#define QWEN_FSCOPE_VNNI 8u\n"
                    "#define QWEN_FSCOPE_AMX  16u\n#define QWEN_FSCOPE_APPLE 32u\n"
                    "#define QWEN_FSCOPE_GPU  64u\n#define QWEN_FSCOPE_ALL 127u\n"
                    "typedef struct { const char *name; unsigned scope; } qwen_flag_scope_t;\n"
                    "static const qwen_flag_scope_t g_qwen_flag_scope[] = {\n")
            for r in rows:
                v = 0
                for fam in r["families"]:
                    v |= bits.get(fam, 0)
                if not v:
                    v = 127
                f.write('    { "%s", %uu },\n' % (r["flag"], v))
            f.write("};\n#endif\n")
        print("wrote %s (%d flags)" % (a.emit_c, len(rows)))

    if a.json:
        json.dump({"schema": 1, "families": FAMILIES, "flags": rows}, open(a.json, "w"), indent=1)
        print("wrote %s (%d flags)" % (a.json, len(rows)))

    by_scope = {}
    for r in rows:
        by_scope.setdefault(r["scope"], []).append(r["flag"])
    print("FLAG PARITY — backend reach derived from the preprocessor guards around each read")
    print("  %d flags read by the engine" % len(rows))
    for s in ("all", "arm", "x86", "apple", "gpu", "none"):
        if s in by_scope:
            print("    %-6s %3d" % (s, len(by_scope[s])))
    print("\n  ARM-only flags (no x86 read site): %s" % ", ".join(by_scope.get("arm", [])) or "none")
    print("\n  x86-only flags (no ARM read site): %s" % ", ".join(by_scope.get("x86", [])) or "none")
    if problems:
        print("\n  DISAGREEMENTS between the documented ISA column and the code:")
        for p in problems:
            print("    %-34s doc=%-6s code=%-5s  %s" %
                  (p["flag"], p["documented_isa"], p["scope"], p["verdict"]))
    else:
        print("\n  no documented-ISA disagreements")
    if a.check and problems:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
