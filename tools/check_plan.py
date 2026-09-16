#!/usr/bin/env python3
"""check_plan.py — PLAN.md must never contain dangling references (ENGINEERING.md, Plan integrity).

    python3 tools/check_plan.py [--plan PLAN.md] [--work .work]

Exit 1 when any of these holds:
  * a `.work/...` path referenced by the plan does not exist;
  * a repository path referenced by the plan (docs/, tools/, tests/, configs/, or a
    top-level *.md) does not exist;
  * two tasks carry the same id (P0.1, P3.3a, ...);
  * a task marked `[x]` references a detail/evidence file that is missing;
  * an addendum under .work/ names a task id (`Task: P1.4`) that PLAN.md does not contain.
A missing PLAN.md is not a failure (the plan is local and untracked): the check reports
it and exits 0.  Nothing here reads the content of the plan beyond paths and task ids.
"""
import argparse, os, re, sys

TASK_RE = re.compile(r"^\s*-\s*\[( |x|X)\]\s+(P\d+\.\d+[a-z]?|[A-Z]+-\d+)\b")
PATH_RE = re.compile(r"(?<![\w/.-])(\.work/[\w./-]+|(?:docs|tools|tests|configs)/[\w./-]+|[\w-]+\.md)(?![\w/-])")
ADDENDUM_TASK_RE = re.compile(r"^\s*(?:[-*]\s*)?Task:\s*\**\s*(P\d+\.\d+[a-z]?|[A-Z]+-\d+)", re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="PLAN.md")
    ap.add_argument("--work", default=".work")
    a = ap.parse_args()
    root = os.getcwd()
    if not os.path.exists(a.plan):
        print(f"check_plan: {a.plan} absent (local, untracked) — nothing to check")
        return 0

    lines = open(a.plan, encoding="utf-8").read().splitlines()
    failures, ids, task_ids = [], {}, set()
    current = None          # (id, done)
    for n, line in enumerate(lines, 1):
        m = TASK_RE.match(line)
        if m:
            done, tid = m.group(1).lower() == "x", m.group(2)
            if tid in ids:
                failures.append(f"line {n}: duplicate task id {tid} (first at line {ids[tid]})")
            ids.setdefault(tid, n)
            task_ids.add(tid)
            current = (tid, done)
        elif line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            current = None  # a heading or paragraph ends the task block
        for p in PATH_RE.findall(line):
            p = p.rstrip(".,;:)")
            if p.endswith(".md") and "/" not in p and not os.path.exists(p):
                # a bare *.md that is not a file here may be prose (e.g. "the runbook.md");
                # only flag it when the line says detail:/doc:/see:
                if not re.search(r"\b(detail|doc|see|evidence):", line):
                    continue
            if not os.path.exists(os.path.join(root, p)):
                who = f" (task {current[0]}{', marked [x]' if current[1] else ''})" if current else ""
                failures.append(f"line {n}: missing path {p}{who}")

    # addenda that name a task the plan does not have
    if os.path.isdir(a.work):
        for dp, _, fs in os.walk(a.work):
            for f in fs:
                if not f.endswith(".md"):
                    continue
                path = os.path.join(dp, f)
                try:
                    txt = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for tid in ADDENDUM_TASK_RE.findall(txt):
                    if tid not in task_ids:
                        failures.append(f"{path}: names task {tid} which is not in {a.plan}")

    n_tasks = len(task_ids)
    n_done = sum(1 for l in lines if TASK_RE.match(l) and TASK_RE.match(l).group(1).lower() == "x")
    if failures:
        print(f"check_plan: FAIL ({len(failures)} problem(s), {n_tasks} tasks, {n_done} done)")
        for f in failures:
            print("  " + f)
        return 1
    print(f"check_plan: PASS ({n_tasks} tasks, {n_done} done, {len(lines)} lines, every referenced path exists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
