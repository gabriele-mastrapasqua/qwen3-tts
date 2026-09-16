#!/usr/bin/env python3
"""check_repo_integrity.py — tracked scripts must not depend on untracked local files.

    python3 tools/check_repo_integrity.py [--strict]

Scans the tracked Makefile, bench.sh, tools/**.{sh,py} and tests/*.{sh,py} for references
to repository-local tooling files (paths under tools/, tests/, configs/, docs/ ending in
.py .sh .json .txt .c .h .md) and reports every reference that

  * exists in this working tree but is NOT tracked by git   -> a fresh clone breaks (FAIL)
  * does not exist at all                                  -> warning (FAIL with --strict);
    such a path may be a documented run output, so it is not an error by itself.

Paths containing shell variables or globs are skipped.  Build products (.o .d .a), virtual
environments and result directories are outside the extension filter on purpose: this is a
dependency check for tooling, not a build-system audit.  Exit 1 on any FAIL.
"""
import argparse, os, re, subprocess, sys

REF_RE = re.compile(r"(?<![\w/.-])((?:tools|tests|configs|docs)/[\w./-]+\.(?:py|sh|json|txt|c|h|md))(?![\w/-])")
SCAN_GLOBS = ["Makefile", "bench.sh", "tools/*.sh", "tools/*.py", "tools/*/*.sh", "tools/*/*.py",
              "tests/*.sh", "tests/*.py"]


def tracked_files():
    out = subprocess.check_output(["git", "ls-files", "-z"]).decode()
    return set(p for p in out.split("\0") if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="missing paths fail too")
    a = ap.parse_args()
    tracked = tracked_files()
    scanned = sorted(p for p in subprocess.check_output(["git", "ls-files", "-z", "--"] + SCAN_GLOBS)
                     .decode().split("\0") if p)
    untracked, missing = {}, {}
    for f in scanned:
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for line in txt.splitlines():
            for ref in REF_RE.findall(line):
                if "$" in ref or "*" in ref or "%" in ref:
                    continue
                if ref in tracked:
                    continue
                (untracked if os.path.exists(ref) else missing).setdefault(ref, set()).add(f)
    for ref, users in sorted(untracked.items()):
        print(f"FAIL  untracked dependency {ref}   <- {', '.join(sorted(users))}")
    for ref, users in sorted(missing.items()):
        print(f"{'FAIL' if a.strict else 'WARN'}  missing path {ref}   <- {', '.join(sorted(users))}")
    bad = len(untracked) + (len(missing) if a.strict else 0)
    print(f"check_repo_integrity: {'FAIL' if bad else 'PASS'} ({len(scanned)} scripts scanned, "
          f"{len(untracked)} untracked, {len(missing)} missing)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
