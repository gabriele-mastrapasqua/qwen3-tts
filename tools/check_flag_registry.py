#!/usr/bin/env python3
"""check_flag_registry.py — every runtime flag the engine reads must also be declared.

A flag that changes what the engine does but never appears in the [FLAGS] line cannot be
verified by perf_profile.py check-flags, which means a deployment can be silently running a
different configuration from the one its profile claims. This test makes that impossible to
introduce by accident: it compares the QWEN_* names the sources read against
g_qwen_reported_flags[] in qwen_tts_kernels.c.

Every QWEN_* string literal in the C sources is an environment variable name — the engine reads
them with getenv() directly, or through the gate table and the *_NCHUNK helpers, which take the
name as a literal argument. So the literals ARE the ground truth.
"""
import glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_FILE = os.path.join(ROOT, "qwen_tts_kernels.c")
REG_RE = re.compile(r'static const char \*const g_qwen_reported_flags\[\] = \{(.*?)\n\};', re.S)


def registry(src):
    m = REG_RE.search(src)
    if not m:
        sys.exit("FAIL: g_qwen_reported_flags[] not found in qwen_tts_kernels.c")
    return set(re.findall(r'"(QWEN_[A-Z0-9_]*[A-Z0-9])"', m.group(1)))


def read_by_engine():
    names = {}
    # every engine source that can call getenv: C, headers, CUDA and Objective-C (Metal)
    for path in sorted(sum((glob.glob(os.path.join(ROOT, ext)) for ext in ("*.c", "*.h", "*.cu", "*.m")), [])):
        src = open(path, errors="replace").read()
        if os.path.basename(path) == "qwen_tts_kernels.c":
            src = REG_RE.sub("", src)          # the register itself is not a read
        for name in re.findall(r'"(QWEN_[A-Z0-9_]*[A-Z0-9])"', src):
            names.setdefault(name, set()).add(os.path.basename(path))
    return names


def main():
    src = open(REG_FILE, errors="replace").read()
    declared = registry(src)
    read = read_by_engine()

    undeclared = sorted(set(read) - declared)
    stale = sorted(declared - set(read))

    print("flags read by the engine: %d · declared in [FLAGS]: %d" % (len(read), len(declared)))
    if undeclared:
        print("\nFAIL: read but never declared — these cannot be verified in a run:")
        for n in undeclared:
            print("   %-32s read in %s" % (n, ", ".join(sorted(read[n]))))
    if stale:
        print("\nFAIL: declared but no longer read — the register advertises a flag that does nothing:")
        for n in stale:
            print("   %s" % n)
    if undeclared or stale:
        print("\nAdd the name to g_qwen_reported_flags[] in qwen_tts_kernels.c, in the group it "
              "belongs to, or remove it there if the flag is gone.")
        return 1
    print("PASS: every flag the engine reads is one it declares")
    return 0


if __name__ == "__main__":
    sys.exit(main())
