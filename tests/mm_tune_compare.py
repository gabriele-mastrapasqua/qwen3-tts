#!/usr/bin/env python3
"""Confronta le soglie del dispatcher fra MACCHINE DIVERSE — x86 e ARM insieme."""
import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GLOB = os.path.join(REPO, "docs", "boxes", "*_tune.json")

def load(path):
    with open(path) as f:
        d = json.load(f)
    d["_file"] = os.path.basename(path)
    stem = d["_file"].replace("_tune.json", "")
    parts = stem.split("_")
    d["_box"] = "_".join(parts[1:]) if len(parts) > 1 else stem
    return d

def cell_key(c):
    return (c["format"], c["rows"], c["cols"], c["threads"])

def winners(d):
    """For each (format, shape, threads) the kernel with the lowest crossover_B > 0.

    crossover_B = 0 nel JSON significa 'non vince mai', ed e' un dato utile: dice che su
    that shape stays on B x matvec on that machine. It is not a gap.
    """
    best = {}
    for c in d.get("cells", []):
        if not c.get("ran"):
            continue
        xb = c.get("crossover_B") or 0
        if xb <= 0:
            best.setdefault(cell_key(c), None)
            continue
        k = cell_key(c)
        cur = best.get(k)
        if cur is None or xb < cur[1]:
            best[k] = (c["kernel"], xb)
    return best

def short(kernel):
    """Nome corto e confrontabile fra ISA: e' il nome che si legge in una tabella."""
    k = kernel.lower()
    for needle, tag in (("amx", "AMX"), ("vnni", "VNNI"), ("avx2", "AVX2"),
                        ("smmla", "SMMLA"), ("bfmmla", "BFMMLA"), ("sdot", "SDOT"),
                        ("tail:", "twin")):
        if needle in k:
            return tag
    return kernel.split()[0][:6]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="tune JSON (default: docs/boxes/*_tune.json)")
    ap.add_argument("--format", dest="fmt", help="this format only (int8/q4/bf16)")
    ap.add_argument("--threads", type=int, help="this thread count only")
    ap.add_argument("--json", action="store_true", help="aggregato in JSON")
    a = ap.parse_args()

    paths = a.files or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"no tune found in {DEFAULT_GLOB}\n"
              f"  prendine uno sul box:  ./qwen_tts --matmat-tune -d <modello> > /tmp/tune.json",
              file=sys.stderr)
        return 1
    boxes = [load(p) for p in paths]

    print("=" * 78)
    print("  SOGLIE DEL DISPATCHER, CONFRONTATE FRA ARCHITETTURE")
    print("=" * 78)
    for d in boxes:
        print(f"  {d['_box']:<28} thread pieni {d.get('threads_full', '?'):<3} "
              f"· {d.get('generated_utc', '?')}")
        src = (d.get("shapes_source") or "")[:100]
        if src:
            print(f"     forme da: {src}")
    print()

    allw = {d["_box"]: winners(d) for d in boxes}
    keys = sorted({k for w in allw.values() for k in w},
                  key=lambda k: (k[0], -(k[1] * k[2]), k[3]))
    if a.fmt:
        keys = [k for k in keys if k[0] == a.fmt]
    if a.threads:
        keys = [k for k in keys if k[3] == a.threads]

    names = [d["_box"] for d in boxes]
    w = max(12, max((len(n) for n in names), default=12))
    print(f"  {'formato':<6} {'forma':<12} {'-j':>3}  " + "".join(f"{n[:w]:<{w}}" for n in names))
    print("  " + "-" * (24 + w * len(names)))
    for k in keys:
        fmt, rows, cols, th = k
        row = f"  {fmt:<6} {f'{rows}x{cols}':<12} {th:>3}  "
        for n in names:
            v = allw[n].get(k)
            row += f"{(f'{short(v[0])}@{v[1]}' if v else '—'):<{w}}"
        print(row)

    print()
    print("  legenda: KERNEL@B = da quel B in su il kernel batchato vince · — = mai,")
    print("           that shape stays on B x matvec on that machine.")
    print("  ⚠️ un kernel che vince SOLO a thread pieni non sta vincendo: sta ammortizzando")
    print("     the pool launches. Read the -j 1 column first.")
    print()

    for d in boxes:
        rec = d.get("recommend") or []
        lines = [r.get("line") for r in rec if r.get("line")]
        if lines:
            print(f"  ── da esportare su {d['_box']}:")
            for l in lines:
                print(f"       {l}")
    if a.json:
        print(json.dumps({n: {f"{k[0]}|{k[1]}x{k[2]}|j{k[3]}": (v[0], v[1]) if v else None
                              for k, v in w.items()} for n, w in allw.items()}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
