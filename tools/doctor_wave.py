#!/usr/bin/env python3
"""Run the grid the doctor recommends — sequentially, from ONE file, with no shell chain.

    python3 tools/doctor_wave.py profiles/doctor/LATEST/wave-plan.json
    python3 tools/doctor_wave.py PLAN --dry-run              # print the commands, run nothing
    python3 tools/doctor_wave.py PLAN --only 1.7b-4x8-cap2   # a subset, by label (comma list)
    python3 tools/doctor_wave.py PLAN --out DIR              # default: <plan dir>/wave

The plan is data (tools/doctor.py wave_plan): a list of runs, each = model, topology, batch
cap, concurrency levels, server env overrides.  Every run is one tests/serve_parallel_wave.py
invocation with the same fixed arguments `make bench-topo` uses (ryan, seed 42, int8, short
texts), so two plans on two boxes are comparable.  Runs go one after the other in list order;
a model whose directory is missing is SKIPPED and said so; a run that fails does not stop the
next one.  Exit code = number of failed runs.  The summary at the end is the per-level table
of every run plus its playback envelope line, which is the gate the user reads.
"""
import argparse, json, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAVE = os.path.join(ROOT, "tests", "serve_parallel_wave.py")
TABLE_ROW = re.compile(r"^\s*(\d+x\d+)\s+(\d+)\s+")
ENVELOPE = re.compile(r"^\s*(PLAYBACK \(client-observed|FIXED BUFFER:)")
FLAGS = re.compile(r"^\s*flags verified in the engine:")


def wave_cmd(run, plan, out_dir, port, bin_path):
    """One serve_parallel_wave.py command for one run of the plan."""
    cmd = [sys.executable, WAVE, "--model", run["model"], "--bin", bin_path,
           "--speaker", plan.get("speaker", "ryan"), "--seed", str(plan.get("seed", 42)),
           "--precision", plan.get("precision", "int8"),
           "--text-file", plan.get("text_file", "tests/load_texts_en.txt"),
           "--classes", plan.get("classes", "short"),
           "--topo", run["topo"], "--conc", ",".join(str(c) for c in run["conc"]),
           "--waves", str(run.get("waves", plan.get("waves", 1))),
           "--batch-cap", str(run["cap"]), "--label", run["label"],
           "--out", os.path.join(out_dir, run["label"]), "--port", str(port), "--no-crosscheck"]
    prof = run.get("profile", plan.get("profile"))
    if prof:
        cmd += ["--profile", prof]
    else:
        cmd += ["--no-profile", "doctor wave plan without a draft profile"]
    if run.get("env"):
        cmd += ["--server-env", ",".join(f"{k}={v}" for k, v in run["env"].items())]
    return cmd


def summarize(log_text):
    """The lines a reader needs from one wave log: flags, the envelope, the per-level rows."""
    keep = []
    for line in log_text.splitlines():
        if FLAGS.match(line) or ENVELOPE.match(line) or TABLE_ROW.match(line):
            keep.append(line.rstrip()[:230])
    return keep


def run_plan(plan, out_dir, port=9500, only=None, dry_run=False, bin_path="./qwen_tts"):
    os.makedirs(out_dir, exist_ok=True)
    runs = plan["runs"]
    if only:
        want = set(only)
        runs = [r for r in runs if r["label"] in want]
        missing = want - {r["label"] for r in runs}
        if missing:
            print(f"doctor_wave: no such run label(s): {sorted(missing)}", file=sys.stderr)
    failed, summary = 0, []
    print(f"### doctor wave plan: {len(runs)} run(s) · profile={plan.get('profile')} · out={out_dir}")
    for i, run in enumerate(runs, 1):
        cmd = wave_cmd(run, plan, out_dir, port, bin_path)
        head = f"##### RUN {i}/{len(runs)} {run['label']}  model={run['model']} topo={run['topo']} cap={run['cap']} conc={run['conc']} env={run.get('env') or '-'}"
        print(head, flush=True)
        if run.get("why"):
            print(f"      why: {run['why']}")
        if dry_run:
            print("      " + " ".join(cmd))
            continue
        if not os.path.isdir(os.path.join(ROOT, run["model"])):
            print(f"      SKIP: model dir {run['model']} not found (download_model.sh)")
            summary.append((run["label"], ["SKIP: model dir missing"]))
            continue
        t0 = time.time()
        log_path = os.path.join(out_dir, run["label"] + ".log")
        with open(log_path, "w") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        with open(log_path) as f:
            lines = summarize(f.read())
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        if rc != 0:
            failed += 1
        print(f"      {status} in {time.time() - t0:.0f}s · log {os.path.relpath(log_path, ROOT)}")
        for line in lines:
            print("   " + line)
        summary.append((run["label"], [l for l in lines if TABLE_ROW.match(l)] or [status]))
    if not dry_run:
        print("\n### SUMMARY (per level: topo C TTFB50 TTFB95 TTFA50 TTFA95 TTFAmax STRM50 STRM95 TOT50 TOT95 ...)")
        for label, rows in summary:
            for r in rows:
                print(f"  {label:<28} {r.strip()}")
        print(f"### DONE: {len(summary)} run(s), {failed} failed")
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="wave-plan.json written by tools/doctor.py")
    ap.add_argument("--out", default=None, help="artifact dir (default <plan dir>/wave)")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--only", default=None, help="comma-separated run labels")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    with open(a.plan) as f:
        plan = json.load(f)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.plan)), "wave")
    only = [s for s in (a.only or "").split(",") if s]
    return min(run_plan(plan, out, a.port, only, a.dry_run, a.bin), 125)


if __name__ == "__main__":
    sys.exit(main())
