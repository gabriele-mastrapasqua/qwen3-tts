#!/usr/bin/env python3
"""backend_matrix.py — PARITY-1/3 matrix, merged from what each build REPORTS.

Cells come from the engine, not from reading the sources: run `--dispatch-map` with
QWEN_DISPATCH_JSON on every build/host you care about, then merge the JSONs here.  Each cell
carries the three states PARITY-2 demands, taken from the engine's own answers:

    IMPLEMENTED  the kernel/feature is compiled into that build   (compiled)
    SELECTABLE   this host can execute it                         (supported)
    EFFECTIVE    it is the resolved answer right now              (on / resolved)

    tools/backend_matrix.py m1.json vnni.json amx.json --out docs/backend-matrix.md
"""
import argparse, json, os, sys


def load(path):
    d = json.load(open(path))
    label = d.get("simd") or "?"
    cls = d.get("isa_class") or "?"
    name = "%s (%s)" % (cls, label)
    cells = {}
    for g in d.get("gates", []):
        st = ("EFFECTIVE" if g.get("on") else
              "SELECTABLE" if g.get("supported") else
              "IMPLEMENTED" if g.get("compiled") else "—")
        cells[("gate", g["id"])] = st
    for f in d.get("features", []):
        res = (f.get("resolved") or "").strip()
        comp, sup = f.get("compiled"), f.get("supported")
        if comp == "no":
            st = "—"
        elif res in ("ON", "yes"):
            st = "EFFECTIVE"
        elif sup == "yes" or comp == "yes":
            st = "SELECTABLE"
        else:
            st = res or "—"
        cells[("feature", f["id"])] = st
    return name, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()

    cols, allcells = [], {}
    for p in a.json:
        name, cells = load(p)
        cols.append(name)
        allcells[name] = cells
    keys = sorted({k for c in allcells.values() for k in c})

    lines = ["# Backend matrix — merged from each build's own `--dispatch-map`", "",
             "Three states, all reported by the engine: **IMPLEMENTED** compiled in · "
             "**SELECTABLE** this host can run it · **EFFECTIVE** it is what resolves now. "
             "`—` means not compiled into that build. Regenerate with `tools/backend_matrix.py`.",
             "", "| feature | " + " | ".join(cols) + " |",
             "|---|" + "---|" * len(cols)]
    for kind, key in keys:
        row = [allcells[c].get((kind, key), "—") for c in cols]
        if all(r == "—" for r in row):
            continue
        lines.append("| `%s` | %s |" % (key, " | ".join(row)))

    gaps = []
    for kind, key in keys:
        vals = {c: allcells[c].get((kind, key), "—") for c in cols}
        arm = [v for c, v in vals.items() if "arm" in c or "apple" in c]
        x86 = [v for c, v in vals.items() if "x86" in c]
        if arm and x86 and any(v != "—" for v in arm) and all(v == "—" for v in x86):
            gaps.append((key, "on ARM only"))
        if arm and x86 and any(v != "—" for v in x86) and all(v == "—" for v in arm):
            gaps.append((key, "on x86 only"))
    if gaps:
        lines += ["", "## Present on one family only", ""]
        lines += ["- `%s` — %s" % (k, w) for k, w in gaps]

    out = "\n".join(lines) + "\n"
    if a.out:
        open(a.out, "w").write(out)
        print("wrote %s (%d rows, %d columns)" % (a.out, len(keys), len(cols)))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
