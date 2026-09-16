#!/usr/bin/env python3
"""soak_fast.py - adaptive SCREEN that locates the playback knee in minutes, not hours.

Why this exists
---------------
A canonical qualification is a 30-minute closed-loop soak per concurrency point. Walking a
capacity ladder that way costs half a day of instance time, and most of it is spent proving
that points we already know are bad are still bad. This driver runs the SAME harness
(`tests/serve_soak.py`), the same profile and the same KPI definitions at a short duration,
classifies each point against a rule fixed in advance, and stops at the knee. You then run
the 30-minute qualification only at the winning point.

A SCREEN IS NEVER A QUALIFICATION. Short windows cannot assess drift, per-class tails or
resource growth, so `serve_soak.py` will report `PARTIAL` and the per-class KPI will be
under-sampled by construction. Every artifact this driver writes is labelled `screen`. Do
not quote a screen number as a supported operating point, and do not put one in a customer
report: use it only to choose which points deserve the long run.

Classification (documented, and printed with the results so a reader can re-derive it):

  CLEAR    stream_p95 <= --stream-clear   and stall@250 == 0 and stall@500 == 0 and 0 errors
  HEALTHY  stream_p95 <  --stream-healthy and stall@250 <= --stall250-max
                                          and stall@500 <= --stall500-max and 0 errors
  KNEE     anything else

The ladder stops after the first KNEE (`--stop-after-knee`, the default). The recommended
qualification point is the highest concurrency that screened CLEAR, falling back to the
highest HEALTHY when nothing is CLEAR.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_SOAK = os.path.join(HERE, "serve_soak.py")


def parse_ladder(spec):
    """"8,10,11,12" or "8:12" or "8:16:2" -> [8, 10, 11, 12] / [8..12] / [8,10,12,14,16]."""
    spec = spec.strip()
    if ":" in spec:
        parts = [int(p) for p in spec.split(":")]
        if len(parts) == 2:
            lo, hi, step = parts[0], parts[1], 1
        elif len(parts) == 3:
            lo, hi, step = parts
        else:
            raise ValueError(f"bad ladder range {spec!r}")
        if step <= 0:
            raise ValueError("ladder step must be positive")
        return list(range(lo, hi + 1, step))
    return [int(p) for p in spec.split(",") if p.strip()]


def run_point(args, conc, out_dir):
    cmd = [
        sys.executable, SERVE_SOAK,
        "--model", args.model,
        "--bin", args.bin,
        "--bank", args.bank,
        "--speaker", args.speaker,
        "--language", args.language,
        "--concurrency", str(conc),
        "--minutes", str(args.minutes),
        "--warmup-s", str(args.warmup_s),
        "--window-s", str(args.window_s),
        "--min-windows", str(args.min_windows),
        "--min-per-window", str(args.min_per_window),
        "--min-per-class", "1",
        "--prefork", str(args.prefork),
        "--prefork-threads", str(args.prefork_threads),
        "--batch-size", str(args.batch_size),
        "--request-timeout", str(args.request_timeout),
        "--port", str(args.port),
        "--out", out_dir,
    ]
    if args.profile:
        cmd += ["--profile", args.profile]
    else:
        cmd += ["--no-profile", args.no_profile]
    if args.precision:
        cmd += ["--precision", args.precision]
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.soak_args:
        cmd += args.soak_args.split()

    env = dict(os.environ)
    if args.model_alias:
        env["QWEN_SOAK_MODEL_ALIAS"] = args.model_alias
    with open(out_dir + ".log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    # a blocked client can orphan the server; never leave one behind between points
    subprocess.run(["pkill", "-9", "-f", "qwen_tts.*--serve"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(args.settle_s)


def read_point(out_dir):
    f = os.path.join(out_dir, "soak_summary.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    s = d.get("sustained") or {}
    pb = s.get("playback") or {}
    if not s.get("stream"):
        return None
    return {
        "completed": d.get("completed", 0),
        "errors": d.get("errors", 0),
        "rejects": d.get("queue_rejected", 0),
        "timeouts": d.get("request_timeout", 0),
        "stream_p50": s["stream"]["p50"],
        "stream_p95": s["stream"]["p95"],
        "ttfa_p95": s["ttfa"]["p95"],
        "safe_p95": (pb.get("safe_play_start") or {}).get("p95"),
        "max_gap_p95": (pb.get("max_gap") or {}).get("p95"),
        "stall250": pb.get("stall_rate_250", 0.0) * 100.0,
        "stall500": pb.get("stall_rate_500", 0.0) * 100.0,
    }


def classify(m, args):
    if m is None:
        return "RUN_FAILED"
    bad_functional = m["errors"] or m["rejects"] or m["timeouts"]
    if bad_functional:
        return "KNEE"
    if (m["stream_p95"] <= args.stream_clear
            and m["stall250"] == 0.0 and m["stall500"] == 0.0):
        return "CLEAR"
    if (m["stream_p95"] < args.stream_healthy
            and m["stall250"] <= args.stall250_max
            and m["stall500"] <= args.stall500_max):
        return "HEALTHY"
    return "KNEE"


HDR = (f"{'C':>4} {'STREAM p50':>10} {'STREAM p95':>10} {'TTFA p95':>9} "
       f"{'safe p95':>9} {'gap p95':>8} {'st@250':>7} {'st@500':>7} {'done':>6} "
       f"{'e/r/t':>7}  verdict")


def fmt_row(conc, m, verdict):
    if m is None:
        return f"{conc:>4} {'-':>10} {'-':>10} {'-':>9} {'-':>9} {'-':>8} {'-':>7} {'-':>7} {'-':>6} {'-':>7}  {verdict}"
    return (f"{conc:>4} {m['stream_p50']:>10.3f} {m['stream_p95']:>10.3f} "
            f"{m['ttfa_p95']:>9.1f} {(m['safe_p95'] or 0):>9.0f} "
            f"{(m['max_gap_p95'] or 0):>8.3f} {m['stall250']:>7.2f} {m['stall500']:>7.2f} "
            f"{m['completed']:>6} {m['errors']}/{m['rejects']}/{m['timeouts']:>1}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description="Adaptive short SCREEN to locate the playback knee.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-alias", default="", help="QWEN_SOAK_MODEL_ALIAS for private checkpoints")
    ap.add_argument("--bin", default="./qwen_tts")
    ap.add_argument("--profile", default="", help="configs/perf profile name or path")
    ap.add_argument("--no-profile", default="fast knee screen with compiled defaults")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--speaker", default="ryan")
    ap.add_argument("--language", default="English")
    ap.add_argument("--precision", default="")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--ladder", required=True, help='"8,10,12" or "8:12" or "8:16:2"')
    ap.add_argument("--prefork", type=int, default=4)
    ap.add_argument("--prefork-threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--request-timeout", type=int, default=600)
    ap.add_argument("--settle-s", type=int, default=3)
    # duration: default 30 s warm-up + 180 s measured as two 90 s windows
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--warmup-s", type=int, default=30)
    ap.add_argument("--window-s", type=int, default=90)
    ap.add_argument("--min-windows", type=int, default=2)
    ap.add_argument("--min-per-window", type=int, default=20)
    # classification rule
    ap.add_argument("--stream-clear", type=float, default=0.90)
    ap.add_argument("--stream-healthy", type=float, default=1.00)
    ap.add_argument("--stall250-max", type=float, default=1.0, help="percent")
    ap.add_argument("--stall500-max", type=float, default=0.0, help="percent")
    ap.add_argument("--stop-after-knee", dest="stop_after_knee", action="store_true", default=True)
    ap.add_argument("--no-stop-after-knee", dest="stop_after_knee", action="store_false",
                    help="walk the whole ladder even past the knee (fuller curve, more instance time)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--soak-args", default="", help="extra args passed through to serve_soak.py")
    args = ap.parse_args()

    ladder = parse_ladder(args.ladder)
    os.makedirs(args.out, exist_ok=True)

    print("=== SOAK-FAST SCREEN — NOT A QUALIFICATION ===")
    print(f"profile={args.profile or '(none)'} model={args.model_alias or args.model}")
    print(f"ladder={ladder}  warmup={args.warmup_s}s  measured={args.min_windows}x{args.window_s}s")
    print(f"rule: CLEAR stream_p95<={args.stream_clear} & stalls 0 | "
          f"HEALTHY stream_p95<{args.stream_healthy} & st250<={args.stall250_max}% "
          f"& st500<={args.stall500_max}% | else KNEE")
    print(HDR, flush=True)

    results = []
    for conc in ladder:
        out_dir = os.path.join(args.out, f"screen-c{conc}")
        run_point(args, conc, out_dir)
        m = read_point(out_dir)
        verdict = classify(m, args)
        print(fmt_row(conc, m, verdict), flush=True)
        results.append({"concurrency": conc, "verdict": verdict, "metrics": m,
                        "evidence": out_dir})
        if verdict in ("KNEE", "RUN_FAILED") and args.stop_after_knee:
            print(f"  -> stopping the ladder at C{conc} ({verdict})")
            break

    clear = [r["concurrency"] for r in results if r["verdict"] == "CLEAR"]
    healthy = [r["concurrency"] for r in results if r["verdict"] == "HEALTHY"]
    recommend = max(clear) if clear else (max(healthy) if healthy else None)
    knee = next((r["concurrency"] for r in results if r["verdict"] == "KNEE"), None)

    print()
    if recommend is None:
        print("RECOMMENDATION: no point screened healthy — do not spend a 30-minute qualification here.")
    else:
        basis = "CLEAR" if clear else "HEALTHY (nothing reached CLEAR)"
        print(f"RECOMMENDATION: qualify at C{recommend} ({basis}).")
        if knee:
            print(f"                first knee at C{knee}; the ladder above it is not worth qualifying.")
        if not clear:
            print("                nothing screened CLEAR: treat C%d as a soft edge, not a preferred point."
                  % recommend)
    print("These are SCREEN results. Run tests/serve_soak.py at 30 minutes with --strict-kpi")
    print("at the recommended point before quoting it anywhere.")

    summary = {
        "kind": "screen",
        "is_qualification": False,
        "profile": args.profile,
        "model_alias": args.model_alias,
        "ladder": ladder,
        "duration": {"warmup_s": args.warmup_s, "window_s": args.window_s,
                     "min_windows": args.min_windows, "minutes": args.minutes},
        "rule": {"stream_clear": args.stream_clear, "stream_healthy": args.stream_healthy,
                 "stall250_max_pct": args.stall250_max, "stall500_max_pct": args.stall500_max},
        "results": results,
        "recommended_qualification_point": recommend,
        "first_knee": knee,
    }
    with open(os.path.join(args.out, "screen_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'screen_summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
