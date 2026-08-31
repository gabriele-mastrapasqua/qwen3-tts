#!/usr/bin/env python3
"""serve_thread_curve.py — how many cores does ONE B=1 stream actually use?"""
import argparse, os, re, resource, statistics, subprocess, time

TEXT = ("The engine renders speech from text, one frame at a time. "
        "This sentence is long enough to make the measurement stable.")

def parse(log):
    t = open(log, errors="replace").read()
    def g(pat, cast=float):
        m = re.search(pat, t)
        return cast(m.group(1)) if m else None
    return {
        "prefill_ms": g(r"Prefill:\s+(\d+)\s*ms"),
        "talker_msf": g(r"Talker step:\s*\d+\s*ms\s*\(([\d.]+)\s*ms/f"),
        "cp_msf":     g(r"Code Predictor:\s*\d+\s*ms\s*\(([\d.]+)\s*ms/f"),
        "rtf":        g(r"RTF\s+([\d.]+)"),
        "ttfa_ms":    g(r"TTFA:\s+(\d+)\s*ms"),
        "frames":     g(r"Generated\s+(\d+)\s+frames"),
        "audio_s":    g(r"Audio:\s+([\d.]+)s generated"),
    }

def run_one(args, kai, j, out, tag):
    env = dict(os.environ)
    if not kai:
        env["QWEN_NO_KAI_I8"] = "1"
        env["QWEN_NO_KAI_BF16"] = "1"
    cmd = [args.bin, "-d", args.model, "-j", str(j),
           "--text", TEXT, "--seed", "42", "-s", "ryan", "-l", "English",
           "-o", os.path.join(out, tag + ".wav")]
    if args.precision == "int8":
        cmd.insert(3, "--int8")
    log = os.path.join(out, tag + ".log")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.time()
    with open(log, "wb") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    d = parse(log)
    d["rc"] = rc
    d["wall_s"] = wall
    d["cpu_s"] = ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime))
    d["nvcsw"] = after.ru_nvcsw - before.ru_nvcsw
    d["nivcsw"] = after.ru_nivcsw - before.ru_nivcsw
    d["csw_s"] = (d["nvcsw"] + d["nivcsw"]) / wall if wall > 0 else 0
    d["cores"] = d["cpu_s"] / wall if wall > 0 else 0
    d["rss_mb"] = after.ru_maxrss / 1024.0
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/kai_thr")
    ap.add_argument("--threads", default="1,2,4,8,16")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--precision", default="int8", choices=["int8", "bf16"])
    ap.add_argument("--bin", default="./qwen_tts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = [int(x) for x in a.threads.split(",")]

    rows = {}
    for j in js:
        for kai in (False, True):
            cells = []
            for i in range(a.runs):
                tag = f"{'on' if kai else 'off'}_j{j}_{i}"
                cells.append(run_one(a, kai, j, a.out, tag))
            def med(k):
                v = [c[k] for c in cells if c.get(k) is not None]
                return statistics.median(v) if v else float("nan")
            rows[(j, kai)] = {k: med(k) for k in
                              ("prefill_ms", "talker_msf", "cp_msf", "rtf", "ttfa_ms",
                               "wall_s", "cpu_s", "csw_s", "cores", "rss_mb", "frames")}
            print(f"  done j={j} kleidi={kai}", flush=True)

    hdr = (f"{'cell':<10}{'prefill':>9}{'tk ms/f':>9}{'cp ms/f':>9}{'RTF':>7}{'TTFA':>8}"
           f"{'cores':>7}{'csw/s':>9}{'RSS MB':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for j in js:
        for kai in (False, True):
            r = rows[(j, kai)]
            print(f"{('kai' if kai else 'base')+'/j'+str(j):<10}{r['prefill_ms']:>9.0f}"
                  f"{r['talker_msf']:>9.2f}{r['cp_msf']:>9.2f}{r['rtf']:>7.3f}"
                  f"{r['ttfa_ms']:>8.0f}{r['cores']:>7.2f}{r['csw_s']:>9.0f}{r['rss_mb']:>8.0f}")

    print("\nscaling vs j=1 (RTF; >1 means faster than one thread)")
    for kai in (False, True):
        base = rows[(js[0], kai)]["rtf"]
        line = "  " + ("kleidi " if kai else "base   ")
        for j in js:
            line += f"j{j}={base / rows[(j, kai)]['rtf']:.2f}x  "
        print(line)
    print("\nmarginal gain per doubling (RTF ratio between successive j)")
    for kai in (False, True):
        line = "  " + ("kleidi " if kai else "base   ")
        for i in range(1, len(js)):
            a_, b_ = rows[(js[i - 1], kai)]["rtf"], rows[(js[i], kai)]["rtf"]
            line += f"{js[i-1]}->{js[i]}={a_ / b_:.2f}x  "
        print(line)

if __name__ == "__main__":
    main()
